import Foundation
import MLX

/// The layers a stage runs: Llama's, Qwen3's and Qwen3's mixture of experts (see modelTypes).
/// Also the iPad's stage of a two-stage Llama-3.2-1B ring pipeline: the last layers, on this
/// device's GPU. Rank 0 (mac/llama_split.py) sends an int32 [L] header and a
/// [1, L, 2048] float16 hidden state per step; this runs its layers with a KV cache and
/// sends the result back. A header of 0 ends the run.
enum LlamaShard {
    /// The model types whose layers this stage runs: Llama's, Qwen3's (which also norms each
    /// head's queries and keys), Qwen3's mixture of experts, and Qwen3.5's (linear attention
    /// layers with a recurrent state, and attention with a gated output), dense or with experts
    /// and a shared expert
    static let modelTypes: Set<String> = ["llama", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"]

    /// How a module's weights are quantized
    struct Quantization {
        var groupSize = 64
        var bits = 4
    }

    /// A model's shape: Llama-3.2-1B-Instruct-4bit by default, else from its config.json
    struct Config {
        var modelType = "llama"
        var hiddenSize = 2048
        var heads = 32
        var kvHeads = 8
        var headDim = 64
        var normEpsilon: Float = 1e-5
        var quantization = Quantization()
        /// By the stage's name of a module (layers.<i>.<path>), when it differs from the rest:
        /// e.g. a mixture of experts' router at 8 bits. The node says (stage_quantization).
        var moduleQuantization: [String: Quantization] = [:]
        // A mixture of experts
        var expertsPerToken = 0
        var normalizeExpertScores = false
        // Linear attention (Qwen3.5)
        var linearKeyHeads = 0
        var linearValueHeads = 0
        var linearKeyDim = 0
        var linearValueDim = 0
        var convKernel = 4

        init() {}

        init(json: [String: Any]) {
            func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
            modelType = json["model_type"] as? String ?? modelType
            hiddenSize = int("hidden_size") ?? hiddenSize
            heads = int("num_attention_heads") ?? heads
            kvHeads = int("num_key_value_heads") ?? heads
            headDim = int("head_dim") ?? hiddenSize / heads
            normEpsilon = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? normEpsilon
            expertsPerToken = int("num_experts_per_tok") ?? 0
            normalizeExpertScores = (json["norm_topk_prob"] as? NSNumber)?.boolValue ?? false
            linearKeyHeads = int("linear_num_key_heads") ?? 0
            linearValueHeads = int("linear_num_value_heads") ?? 0
            linearKeyDim = int("linear_key_head_dim") ?? 0
            linearValueDim = int("linear_value_head_dim") ?? 0
            convKernel = int("linear_conv_kernel_dim") ?? convKernel
            func parsed(_ entry: [String: Any], else fallback: Quantization) -> Quantization {
                Quantization(
                    groupSize: (entry["group_size"] as? NSNumber)?.intValue ?? fallback.groupSize,
                    bits: (entry["bits"] as? NSNumber)?.intValue ?? fallback.bits)
            }
            if let entries = json["quantization"] as? [String: Any] {
                quantization = parsed(entries, else: quantization)
                for (path, entry) in entries {
                    if let entry = entry as? [String: Any] {
                        moduleQuantization[path] = parsed(entry, else: quantization)
                    }
                }
            }
        }
    }

    struct Layer {
        let weights: [String: MLXArray]
        var config = Config()
        /// By the module's path in this layer, where it differs from config.quantization
        var quantization: [String: Quantization] = [:]

        func weight(_ name: String) -> MLXArray {
            guard let array = weights[name] else { fatalError("missing weight \(name)") }
            return array
        }

        /// A linear attention layer (Qwen3.5), which keeps a recurrent state, not keys and values
        var isLinear: Bool { weights["linear_attn.A_log"] != nil }

        func quantization(of module: String) -> Quantization {
            quantization[module] ?? config.quantization
        }

        func linear(_ x: MLXArray, _ name: String) -> MLXArray {
            let quantization = quantization(of: name)
            return quantizedMatmul(
                x, weight(name + ".weight"), scales: weight(name + ".scales"),
                biases: weight(name + ".biases"), transpose: true,
                groupSize: quantization.groupSize, bits: quantization.bits)
        }

        /// One expert's projection of each row, by `experts` (the expert of each row)
        func expertLinear(_ x: MLXArray, _ name: String, experts: MLXArray, sorted: Bool) -> MLXArray {
            let quantization = quantization(of: name)
            return gatherQuantizedMM(
                x, weight(name + ".weight"), scales: weight(name + ".scales"),
                biases: weight(name + ".biases"), rhsIndices: experts, transpose: true,
                groupSize: quantization.groupSize, bits: quantization.bits,
                sortedIndices: sorted)
        }
    }

    /// A layer's cache, as attention uses it: where RoPE starts, the mask, and the keys and
    /// values to attend over once this step's are added
    protocol LayerCache {
        func rope(_ x: MLXArray, headDim: Int, freqs: MLXArray) -> MLXArray
        func mask(_ length: Int) -> MLXFast.ScaledDotProductAttentionMaskMode
        mutating func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    }

    struct KVCache: LayerCache {
        var keys: MLXArray?
        var values: MLXArray?

        /// Tokens held: keys and values are kept at exactly this length
        var offset: Int { keys?.dim(2) ?? 0 }

        /// Drop every token from `offset` on
        func truncated(to offset: Int) -> KVCache {
            guard let keys, let values, offset < self.offset else { return self }
            if offset == 0 { return KVCache() }
            return KVCache(keys: keys[.ellipsis, ..<offset, 0...], values: values[.ellipsis, ..<offset, 0...])
        }

        func rope(_ x: MLXArray, headDim: Int, freqs: MLXArray) -> MLXArray {
            // As many of each head's dimensions as there are frequencies for (Qwen3.5 rotates a
            // quarter of them)
            MLXFast.RoPE(
                x, dimensions: freqs.dim(0) * 2, traditional: false, base: nil, scale: 1,
                offset: offset, freqs: freqs)
        }

        func mask(_ length: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
            length > 1 ? .causal : .none
        }

        mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
            if let keys, let values {
                self.keys = concatenated([keys, newKeys], axis: 2)
                self.values = concatenated([values, newValues], axis: 2)
            } else {
                keys = newKeys
                values = newValues
            }
            return (keys!, values!)
        }
    }

    /// mlx-lm's BatchKVCache: rows left-padded to one length, each with its own offset. Keys
    /// and values are kept at exactly `length` (mlx-lm allocates in steps of 256; past
    /// `length` its buffer holds nothing any mask lets through, so the two agree).
    struct BatchKVCache: LayerCache {
        var keys: MLXArray?
        var values: MLXArray?
        var leftPadding: [Int]
        var offsets: [Int]
        var length = 0
        var rightPadding: [Int]?

        init(leftPadding: [Int]) {
            self.leftPadding = leftPadding
            offsets = leftPadding.map { -$0 }
        }

        static func merge(_ caches: [KVCache]) -> BatchKVCache {
            let lengths = caches.map(\.offset)
            let longest = lengths.max() ?? 0
            var batch = BatchKVCache(leftPadding: lengths.map { longest - $0 })
            guard longest > 0, let template = caches.first(where: { $0.keys != nil })?.keys,
                let valueTemplate = caches.first(where: { $0.values != nil })?.values
            else { return batch }
            func padded(_ array: MLXArray?, like template: MLXArray, by padding: Int) -> MLXArray {
                let shape = [1, template.dim(1), padding, template.dim(3)]
                guard let array else {
                    return MLXArray.zeros([1, template.dim(1), longest, template.dim(3)], dtype: template.dtype)
                }
                return padding == 0 ? array : concatenated([MLXArray.zeros(shape, dtype: template.dtype), array], axis: 2)
            }
            batch.keys = concatenated(
                zip(caches, batch.leftPadding).map { padded($0.keys, like: template, by: $1) }, axis: 0)
            batch.values = concatenated(
                zip(caches, batch.leftPadding).map { padded($0.values, like: valueTemplate, by: $1) }, axis: 0)
            batch.offsets = batch.offsets.map { $0 + longest }
            batch.length = longest
            return batch
        }

        func rope(_ x: MLXArray, headDim: Int, freqs: MLXArray) -> MLXArray {
            MLXFast.RoPE(
                x, dimensions: freqs.dim(0) * 2, traditional: false, base: nil, scale: 1,
                offset: MLXArray(offsets.map(Int32.init)), freqs: freqs)
        }

        /// mlx-lm's create_causal_mask(N, offset: length, left_padding:)
        func mask(_ count: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
            let columns = MLXArray(Int32(0) ..< Int32(length + count))
            let rows = MLXArray(Int32(length) ..< Int32(length + count))
            var mask = rows.expandedDimensions(axis: 1) .>= columns.expandedDimensions(axis: 0)
            let padding = MLXArray(leftPadding.map(Int32.init)).reshaped(-1, 1, 1, 1)
            mask = mask .&& (padding .<= columns.reshaped(1, 1, 1, -1))
            return .array(mask)
        }

        mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
            if let keys, let values {
                self.keys = concatenated([keys, newKeys], axis: 2)
                self.values = concatenated([values, newValues], axis: 2)
            } else {
                keys = newKeys
                values = newValues
            }
            length += newKeys.dim(2)
            offsets = offsets.map { $0 + newKeys.dim(2) }
            return (keys!, values!)
        }

        mutating func prepare(leftPadding added: [Int], rightPadding: [Int]) {
            if !added.isEmpty {
                leftPadding = zip(leftPadding, added).map(+)
                offsets = zip(offsets, added).map(-)
            }
            if (rightPadding.max() ?? 0) > 0 { self.rightPadding = rightPadding }
        }

        /// Turn right padding into left padding: roll each row right by its padding
        mutating func finalize() {
            guard let padding = rightPadding else { return }
            if let keys, let values {
                func rolled(_ array: MLXArray) -> MLXArray {
                    concatenated(padding.enumerated().map { row, shift in
                        let line = array[row ..< row + 1]
                        guard shift > 0 else { return line }
                        let total = line.dim(2)
                        return concatenated(
                            [line[.ellipsis, (total - shift)..., 0...], line[.ellipsis, ..<(total - shift), 0...]],
                            axis: 2)
                    }, axis: 0)
                }
                self.keys = rolled(keys)
                self.values = rolled(values)
            }
            offsets = zip(offsets, padding).map(-)
            leftPadding = zip(leftPadding, padding).map(+)
            rightPadding = nil
        }

        mutating func trim(_ count: Int) {
            let dropped = min(length, count)
            length -= dropped
            offsets = offsets.map { $0 - dropped }
            if let keys, let values {
                self.keys = keys[.ellipsis, ..<length, 0...]
                self.values = values[.ellipsis, ..<length, 0...]
            }
        }

        mutating func filter(_ rows: [Int]) {
            if let keys, let values, !rows.isEmpty {
                let index = MLXArray(rows.map(Int32.init))
                self.keys = keys[index]
                self.values = values[index]
            } else if rows.isEmpty {
                keys = nil
                values = nil
            }
            offsets = rows.map { offsets[$0] }
            leftPadding = rows.map { leftPadding[$0] }
            // Drop the padding every row has
            if let least = leftPadding.min(), least > 0 {
                if let keys, let values {
                    self.keys = keys[.ellipsis, least..., 0...]
                    self.values = values[.ellipsis, least..., 0...]
                }
                length -= least
                leftPadding = leftPadding.map { $0 - least }
            }
        }

        mutating func extend(_ other: BatchKVCache) {
            if keys == nil && other.keys == nil {
                leftPadding += other.leftPadding
                offsets += other.offsets
                return
            }
            let longest = max(length, other.length)
            let template = (keys ?? other.keys)!
            let valueTemplate = (values ?? other.values)!
            func padded(_ cache: BatchKVCache) -> (MLXArray, MLXArray, [Int]) {
                let rows = cache.offsets.count
                let left = longest - cache.length
                var k = cache.keys ?? MLXArray.zeros([rows, template.dim(1), 0, template.dim(3)], dtype: template.dtype)
                var v = cache.values ?? MLXArray.zeros([rows, valueTemplate.dim(1), 0, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
                if left > 0 {
                    k = concatenated([MLXArray.zeros([rows, k.dim(1), left, k.dim(3)], dtype: k.dtype), k], axis: 2)
                    v = concatenated([MLXArray.zeros([rows, v.dim(1), left, v.dim(3)], dtype: v.dtype), v], axis: 2)
                }
                return (k, v, cache.leftPadding.map { $0 + left })
            }
            let (k1, v1, p1) = padded(self)
            let (k2, v2, p2) = padded(other)
            keys = concatenated([k1, k2], axis: 0)
            values = concatenated([v1, v2], axis: 0)
            leftPadding = p1 + p2
            offsets += other.offsets
            length = longest
        }

        func extract(_ row: Int) -> KVCache {
            guard let keys, let values else { return KVCache() }
            let padding = leftPadding[row]
            return KVCache(
                keys: keys[row ..< row + 1, 0..., padding ..< length, 0...],
                values: values[row ..< row + 1, 0..., padding ..< length, 0...])
        }
    }

    /// This stage's layers and the llama3 RoPE frequencies, from a shard made by mac/make_shard.py
    static func loadLayers(shard: URL) throws -> (layers: [Layer], ropeFrequencies: MLXArray) {
        try loadLayers(arrays: try loadArrays(url: shard), config: Config())
    }

    /// Layers named layers.0.., plus rope_freqs
    static func loadLayers(arrays: [String: MLXArray], config: Config) throws -> (layers: [Layer], ropeFrequencies: MLXArray) {
        guard let ropeFrequencies = arrays["rope_freqs"] else { fatalError("shard has no rope_freqs") }
        var layers: [Layer] = []
        while true {
            let prefix = "layers.\(layers.count)."
            let layerWeights = arrays.filter { $0.key.hasPrefix(prefix) }
            if layerWeights.isEmpty { break }
            func ownNames<Value>(_ entries: [String: Value]) -> [String: Value] {
                Dictionary(uniqueKeysWithValues: entries.filter { $0.key.hasPrefix(prefix) }.map {
                    (String($0.key.dropFirst(prefix.count)), $0.value)
                })
            }
            layers.append(Layer(
                weights: ownNames(layerWeights), config: config,
                quantization: ownNames(config.moduleQuantization)))
        }
        eval(Array(arrays.values))
        return (layers, ropeFrequencies)
    }

    /// An attention layer, given its cache
    static func forward<Cache: LayerCache>(
        _ x: MLXArray, layer: Layer, cache: inout Cache, ropeFrequencies: MLXArray
    ) -> MLXArray {
        block(x, layer: layer) { attention($0, layer: layer, cache: &cache, ropeFrequencies: ropeFrequencies) }
    }

    /// A linear attention layer, given its cache
    static func forward(_ x: MLXArray, layer: Layer, cache: inout RecurrentCache) -> MLXArray {
        block(x, layer: layer) { linearAttention($0, layer: layer, cache: &cache) }
    }

    /// A decoder layer around its attention (`mix`): normed, mixed, then its MLP or experts
    static func block(_ x: MLXArray, layer: Layer, mix: (MLXArray) -> MLXArray) -> MLXArray {
        let normEpsilon = layer.config.normEpsilon
        let normed = MLXFast.rmsNorm(x, weight: layer.weight("input_layernorm.weight"), eps: normEpsilon)
        let attended = x + mix(normed)
        let normedAgain = MLXFast.rmsNorm(
            attended, weight: layer.weight("post_attention_layernorm.weight"), eps: normEpsilon)
        if layer.weights["mlp.switch_mlp.gate_proj.weight"] != nil {
            return attended + mixtureOfExperts(normedAgain, layer: layer)
        }
        let gate = layer.linear(normedAgain, "mlp.gate_proj")
        let up = layer.linear(normedAgain, "mlp.up_proj")
        return attended + layer.linear(gate * sigmoid(gate) * up, "mlp.down_proj")
    }

    static func attention<Cache: LayerCache>(
        _ x: MLXArray, layer: Layer, cache: inout Cache, ropeFrequencies: MLXArray
    ) -> MLXArray {
        let (heads, kvHeads, headDim, normEpsilon) = (
            layer.config.heads, layer.config.kvHeads, layer.config.headDim, layer.config.normEpsilon)
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = layer.linear(x, "self_attn.q_proj")
        // Qwen3.5's attention gates its output: q_proj makes each head's queries and gate
        var gate: MLXArray? = nil
        if queries.dim(-1) == 2 * heads * headDim {
            let parts = split(queries.reshaped(batch, length, heads, 2 * headDim), parts: 2, axis: -1)
            queries = parts[0]
            gate = parts[1].reshaped(batch, length, heads * headDim)
        } else {
            queries = queries.reshaped(batch, length, heads, headDim)
        }
        var keys = layer.linear(x, "self_attn.k_proj").reshaped(batch, length, kvHeads, headDim)
        var values = layer.linear(x, "self_attn.v_proj")
            .reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        // Qwen3 norms each head's queries and keys before RoPE
        if let queryNorm = layer.weights["self_attn.q_norm.weight"],
            let keyNorm = layer.weights["self_attn.k_norm.weight"]
        {
            queries = MLXFast.rmsNorm(queries, weight: queryNorm, eps: normEpsilon)
            keys = MLXFast.rmsNorm(keys, weight: keyNorm, eps: normEpsilon)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

        queries = cache.rope(queries, headDim: headDim, freqs: ropeFrequencies)
        keys = cache.rope(keys, headDim: headDim, freqs: ropeFrequencies)
        let mask = cache.mask(length)
        (keys, values) = cache.update(keys: keys, values: values)

        var attention = attend(
            queries: queries, keys: keys, values: values, scale: 1 / Float(headDim).squareRoot(),
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim)
        if let gate { attention = attention * sigmoid(gate) }
        return layer.linear(attention, "self_attn.o_proj")
    }

    /// Scaled dot-product attention. For heads 256 wide (Qwen3.5) over more than 8 queries,
    /// MLX's own unfused fallback, op for op: on a GPU with neural accelerators (NAX) MLX
    /// 0.32.4 picks a kernel for 1024 queries or more that fails to compile there ("Unable to
    /// build metal library from source", attention_nax_dsplit), and below that, and on a Mac
    /// without NAX below 2048 queries, MLX runs this fallback anyway.
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let headDim = queries.dim(-1)
        guard headDim == 256 && queries.dim(2) > 8 else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        let (queryHeads, kvHeads) = (queries.dim(1), keys.dim(1))
        let repeats = queryHeads / kvHeads
        var q = MLXArray(scale).asType(queries.dtype) * queries
        var k = keys
        var v = values
        if repeats > 1 {
            q = unflatten(q, axis: 1, shape: [kvHeads, repeats])
            k = expandedDimensions(k, axis: 2)
            v = expandedDimensions(v, axis: 2)
        }
        var scores = matmul(q, k.swappedAxes(-1, -2))
        var maskArray: MLXArray? = nil
        switch mask {
        case .causal:
            let (keyLength, queryLength) = (k.dim(-2), q.dim(-2))
            let offset = keyLength - queryLength
            maskArray = MLXArray(Int32(offset) ..< Int32(queryLength + offset)).expandedDimensions(axis: 1)
                .>= MLXArray(Int32(0) ..< Int32(keyLength)).expandedDimensions(axis: 0)
        case .array(let given):
            maskArray = given
        default:
            break
        }
        if var maskArray {
            if repeats > 1 && maskArray.ndim >= 3 {
                maskArray = maskArray.dim(-3) == 1
                    ? expandedDimensions(maskArray, axis: -3)
                    : unflatten(maskArray, axis: -3, shape: [kvHeads, repeats])
            }
            if maskArray.dtype == .bool {
                scores = which(maskArray, scores, MLXArray(lowest(scores.dtype)).asType(scores.dtype))
            } else {
                scores = scores + maskArray
            }
        }
        scores = softmax(scores, axis: -1, precise: true)
        var out = matmul(scores, v)
        if repeats > 1 {
            out = out.flattened(start: 1, end: 2)
        }
        return out
    }

    /// finfo(dtype).min
    static func lowest(_ dtype: DType) -> Float {
        switch dtype {
        case .float16: -65504
        case .bfloat16: -3.3895313892515355e38
        default: -Float.greatestFiniteMagnitude
        }
    }

    /// Qwen3's mixture of experts, as mlx-lm's Qwen3MoeSparseMoeBlock and SwitchGLU compute it:
    /// each token's top k experts by the router, their outputs weighed by the router's scores
    static func mixtureOfExperts(_ x: MLXArray, layer: Layer) -> MLXArray {
        let k = layer.config.expertsPerToken
        let gates = softmax(layer.linear(x, "mlp.gate"), axis: -1, precise: true)
        let experts = argPartition(gates, kth: -k, axis: -1)[.ellipsis, (-k)...]
        var scores = takeAlong(gates, experts, axis: -1)
        if layer.config.normalizeExpertScores {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        // [B, L, 1, 1, D]: one row per token, for each of its experts to project
        var rows = expandedDimensions(x, axes: [-2, -3])
        var rowExperts = experts
        // With many tokens, sorted by expert, so each expert's weights are read once
        let sorted = experts.size >= 64
        var inverseOrder: MLXArray? = nil
        if sorted {
            let flatExperts = experts.flattened()
            let order = argSort(flatExperts)
            inverseOrder = argSort(order)
            rows = rows.flattened(start: 0, end: -3)[order.floorDivide(k)]
            rowExperts = flatExperts[order]
        }
        let up = layer.expertLinear(rows, "mlp.switch_mlp.up_proj", experts: rowExperts, sorted: sorted)
        let gate = layer.expertLinear(rows, "mlp.switch_mlp.gate_proj", experts: rowExperts, sorted: sorted)
        var projected = layer.expertLinear(
            gate * sigmoid(gate) * up, "mlp.switch_mlp.down_proj", experts: rowExperts, sorted: sorted)
        if let inverseOrder {
            projected = unflatten(projected[inverseOrder], axis: 0, shape: experts.shape)
        }
        // [B, L, k, D], weighed by score and summed over the k experts
        projected = projected.squeezed(axis: -2)
        let routed = (projected * expandedDimensions(scores, axis: -1)).sum(axis: -2).asType(projected.dtype)
        // Qwen3.5's shared expert (Qwen3NextSparseMoeBlock): an MLP every token goes through,
        // weighed by its own gate
        guard layer.weights["mlp.shared_expert.gate_proj.weight"] != nil else { return routed }
        let sharedGate = layer.linear(x, "mlp.shared_expert.gate_proj")
        let shared = layer.linear(sharedGate * sigmoid(sharedGate) * layer.linear(x, "mlp.shared_expert.up_proj"),
            "mlp.shared_expert.down_proj")
        return routed + sigmoid(layer.linear(x, "mlp.shared_expert_gate")) * shared
    }

    static func run(rank: Int, hostfileJSON: String, shard: URL, log: @escaping (String) -> Void) throws {
        let hostfile = FileManager.default.temporaryDirectory.appendingPathComponent("mlx-hostfile.json")
        try hostfileJSON.write(to: hostfile, atomically: true, encoding: .utf8)
        setenv("MLX_HOSTFILE", hostfile.path, 1)
        setenv("MLX_RANK", String(rank), 1)

        // Load this stage's layers
        var started = Date()
        let (layers, ropeFrequencies) = try loadLayers(shard: shard)
        log("loaded \(layers.count) layers in \(seconds(since: started))")

        started = Date()
        let group = try MLXDistributed.initialize(backend: .ring, strict: true)
        let peer = 1 - (try group.rank)
        log("joined ring as rank \(try group.rank) of \(try group.size) in \(seconds(since: started))")

        var caches = Array(repeating: KVCache(), count: layers.count)
        var steps = 0
        var computeSeconds: Double = 0
        while true {
            let header = MLXDistributed.recv([1], dtype: .int32, from: peer, group: group)
            eval(header)
            let length = Int(header.item(Int32.self))
            if length == 0 { break }

            var hidden = MLXDistributed.recv([1, length, Config().hiddenSize], dtype: .float16, from: peer, group: group)
            eval(hidden)
            let computeStarted = Date()
            for index in layers.indices {
                hidden = forward(
                    hidden, layer: layers[index], cache: &caches[index],
                    ropeFrequencies: ropeFrequencies)
            }
            hidden = hidden.asType(.float16)
            eval(hidden)
            if steps > 0 { computeSeconds += Date().timeIntervalSince(computeStarted) }
            eval(MLXDistributed.send(hidden, to: peer, group: group))

            if steps == 0 { log("prefill: \(length) tokens") }
            steps += 1
        }
        log(String(
            format: "done: %d steps; iPad GPU %.2f ms per decode token for %d layers",
            steps, computeSeconds * 1000 / Double(max(steps - 1, 1)), layers.count))
        log("DONE")
    }

    private static func seconds(since start: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(start))
    }
}
