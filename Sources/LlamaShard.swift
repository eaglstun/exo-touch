import Foundation
import MLX

/// The iPad's stage of a two-stage Llama-3.2-1B pipeline: the last layers, on this
/// device's GPU. Rank 0 (mac/llama_split.py) sends an int32 [L] header and a
/// [1, L, 2048] float16 hidden state per step; this runs its layers with a KV cache and
/// sends the result back. A header of 0 ends the run.
enum LlamaShard {
    /// A Llama model's shape: Llama-3.2-1B-Instruct-4bit by default, else from its config.json
    struct Config {
        var hiddenSize = 2048
        var heads = 32
        var kvHeads = 8
        var headDim = 64
        var normEpsilon: Float = 1e-5
        var groupSize = 64
        var bits = 4

        init() {}

        init(json: [String: Any]) {
            func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
            hiddenSize = int("hidden_size") ?? hiddenSize
            heads = int("num_attention_heads") ?? heads
            kvHeads = int("num_key_value_heads") ?? heads
            headDim = int("head_dim") ?? hiddenSize / heads
            normEpsilon = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? normEpsilon
            if let quantization = json["quantization"] as? [String: Any] {
                groupSize = (quantization["group_size"] as? NSNumber)?.intValue ?? groupSize
                bits = (quantization["bits"] as? NSNumber)?.intValue ?? bits
            }
        }
    }

    struct Layer {
        let weights: [String: MLXArray]
        var config = Config()

        func weight(_ name: String) -> MLXArray {
            guard let array = weights[name] else { fatalError("missing weight \(name)") }
            return array
        }

        func linear(_ x: MLXArray, _ name: String) -> MLXArray {
            quantizedMatmul(
                x, weight(name + ".weight"), scales: weight(name + ".scales"),
                biases: weight(name + ".biases"), transpose: true,
                groupSize: config.groupSize, bits: config.bits)
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
            MLXFast.RoPE(
                x, dimensions: headDim, traditional: false, base: nil, scale: 1, offset: offset,
                freqs: freqs)
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
                x, dimensions: headDim, traditional: false, base: nil, scale: 1,
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
            layers.append(Layer(weights: Dictionary(uniqueKeysWithValues: layerWeights.map {
                (String($0.key.dropFirst(prefix.count)), $0.value)
            }), config: config))
        }
        eval(Array(arrays.values))
        return (layers, ropeFrequencies)
    }

    static func forward<Cache: LayerCache>(
        _ x: MLXArray, layer: Layer, cache: inout Cache, ropeFrequencies: MLXArray
    ) -> MLXArray {
        let (heads, kvHeads, headDim, normEpsilon) = (
            layer.config.heads, layer.config.kvHeads, layer.config.headDim, layer.config.normEpsilon)
        let batch = x.dim(0)
        let length = x.dim(1)

        let normed = MLXFast.rmsNorm(x, weight: layer.weight("input_layernorm.weight"), eps: normEpsilon)
        var queries = layer.linear(normed, "self_attn.q_proj")
            .reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = layer.linear(normed, "self_attn.k_proj")
            .reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        var values = layer.linear(normed, "self_attn.v_proj")
            .reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        queries = cache.rope(queries, headDim: headDim, freqs: ropeFrequencies)
        keys = cache.rope(keys, headDim: headDim, freqs: ropeFrequencies)
        let mask = cache.mask(length)
        (keys, values) = cache.update(keys: keys, values: values)

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / Float(headDim).squareRoot(),
            mask: mask)
        let attended = x + layer.linear(
            attention.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim),
            "self_attn.o_proj")

        let normedAgain = MLXFast.rmsNorm(
            attended, weight: layer.weight("post_attention_layernorm.weight"), eps: normEpsilon)
        let gate = layer.linear(normedAgain, "mlp.gate_proj")
        let up = layer.linear(normedAgain, "mlp.up_proj")
        return attended + layer.linear(gate * sigmoid(gate) * up, "mlp.down_proj")
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
