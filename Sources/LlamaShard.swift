import Foundation
import MLX

/// The iPad's stage of a two-stage Llama-3.2-1B pipeline: the last layers, on this
/// device's GPU. Rank 0 (mac/llama_split.py) sends an int32 [L] header and a
/// [1, L, 2048] float16 hidden state per step; this runs its layers with a KV cache and
/// sends the result back. A header of 0 ends the run.
enum LlamaShard {
    static let hiddenSize = 2048
    static let heads = 32
    static let kvHeads = 8
    static let headDim = 64
    static let normEpsilon: Float = 1e-5
    static let groupSize = 64
    static let bits = 4

    struct Layer {
        let weights: [String: MLXArray]

        func weight(_ name: String) -> MLXArray {
            guard let array = weights[name] else { fatalError("missing weight \(name)") }
            return array
        }

        func linear(_ x: MLXArray, _ name: String) -> MLXArray {
            quantizedMatmul(
                x, weight(name + ".weight"), scales: weight(name + ".scales"),
                biases: weight(name + ".biases"), transpose: true,
                groupSize: LlamaShard.groupSize, bits: LlamaShard.bits)
        }
    }

    struct KVCache {
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
    }

    /// This stage's layers and the llama3 RoPE frequencies, from a shard made by mac/make_shard.py
    static func loadLayers(shard: URL) throws -> (layers: [Layer], ropeFrequencies: MLXArray) {
        let arrays = try loadArrays(url: shard)
        guard let ropeFrequencies = arrays["rope_freqs"] else { fatalError("shard has no rope_freqs") }
        var layers: [Layer] = []
        while true {
            let prefix = "layers.\(layers.count)."
            let layerWeights = arrays.filter { $0.key.hasPrefix(prefix) }
            if layerWeights.isEmpty { break }
            layers.append(Layer(weights: Dictionary(uniqueKeysWithValues: layerWeights.map {
                (String($0.key.dropFirst(prefix.count)), $0.value)
            })))
        }
        eval(Array(arrays.values))
        return (layers, ropeFrequencies)
    }

    static func forward(
        _ x: MLXArray, layer: Layer, cache: inout KVCache, ropeFrequencies: MLXArray
    ) -> MLXArray {
        let offset = cache.offset
        let batch = x.dim(0)
        let length = x.dim(1)

        let normed = MLXFast.rmsNorm(x, weight: layer.weight("input_layernorm.weight"), eps: normEpsilon)
        var queries = layer.linear(normed, "self_attn.q_proj")
            .reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = layer.linear(normed, "self_attn.k_proj")
            .reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        var values = layer.linear(normed, "self_attn.v_proj")
            .reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        queries = MLXFast.RoPE(
            queries, dimensions: headDim, traditional: false, base: nil, scale: 1, offset: offset,
            freqs: ropeFrequencies)
        keys = MLXFast.RoPE(
            keys, dimensions: headDim, traditional: false, base: nil, scale: 1, offset: offset,
            freqs: ropeFrequencies)

        if let cachedKeys = cache.keys, let cachedValues = cache.values {
            keys = concatenated([cachedKeys, keys], axis: 2)
            values = concatenated([cachedValues, values], axis: 2)
        }
        cache.keys = keys
        cache.values = values

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / Float(headDim).squareRoot(),
            mask: length > 1 ? .causal : .none)
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

            var hidden = MLXDistributed.recv([1, length, hiddenSize], dtype: .float16, from: peer, group: group)
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
