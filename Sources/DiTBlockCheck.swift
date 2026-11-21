import Foundation
import MLX

/// FastFrames-LingBot's real DiT block 0 on this device, from the inputs, weights and PyTorch output
/// that fastframes' scripts/lingbot_block_capture.py saved during a full-cache chunk: is the MLX-Swift
/// block right, and how fast is it? A port of scripts/lingbot_block_mlx.py (same dtype rules: FP32
/// residual stream and norm statistics, BF16 matmuls with BF16 bias, FP32 RoPE rotations rounded to
/// BF16, this chunk's cache rows overwritten before attention).
///
///     SPIKE_MODE=ditblock SPIKE_BLOCK=<name>: reads Documents/<name>.safetensors and <name>.json
///     (mac/run_dit_block.sh copies them over and runs this)
enum DiTBlockCheck {
    struct Meta: Decodable {
        let size: String
        let end: Int
        let tokens: Int
        let heads: Int
        let eps: Float
        let attn_eps: Float
        let cross_eps: Float
        let norm3_eps: Float
        let torch_block_ms: Double
    }

    static func run(name: String, log: @escaping (String) -> Void) throws {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let meta = try JSONDecoder().decode(
            Meta.self, from: Data(contentsOf: documents.appendingPathComponent("\(name).json")))
        var w = try loadArrays(url: documents.appendingPathComponent("\(name).safetensors"))
        for (key, value) in w { w[key] = value.asType(value.dtype) }  // move off the CPU stream's lazy load
        eval(Array(w.values))
        let t = meta.tokens, h = meta.heads
        let dim = w["value"]!.dim(-1), d = dim / h, end = meta.end
        let scale = 1 / Float(d).squareRoot()
        log("block 0 at \(meta.size): \(t) query tokens, \(end) cache rows; device \(Device.defaultDevice())")

        func linear(_ x: MLXArray, _ weight: String, _ bias: String) -> MLXArray {
            addMM(w[bias]!.asType(.bfloat16), x.asType(.bfloat16), w[weight]!.asType(.bfloat16).T)
        }
        func layerNorm(_ x: MLXArray, _ eps: Float, weight: String? = nil, bias: String? = nil) -> MLXArray {
            MLXFast.layerNorm(x.asType(.float32), weight: weight.map { w[$0]!.asType(.float32) },
                              bias: bias.map { w[$0]!.asType(.float32) }, eps: eps).asType(x.dtype)
        }
        func rmsNorm(_ x: MLXArray, _ weight: String, _ eps: Float) -> MLXArray {
            let xf = x.asType(.float32)
            let normed = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)
            return normed.asType(x.dtype) * w[weight]!
        }
        func ropeReal(_ x: MLXArray) -> MLXArray {  // x [1, T, H, D]; rope [T, 1, D/2, 2] FP32
            let shape = x.shape
            let pairs = x.asType(.float32).reshaped(shape.dropLast() + [d / 2, 2])
            let xr = pairs[.ellipsis, 0], xi = pairs[.ellipsis, 1]
            let table = w["rope"]!
            let fr = table[.ellipsis, 0], fi = table[.ellipsis, 1]
            return stacked([xr * fr - xi * fi, xr * fi + xi * fr], axis: -1).reshaped(shape).asType(x.dtype)
        }
        func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {  // [1, L, H, D] layout
            MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                scale: scale, mask: nil
            ).transposed(0, 2, 1, 3)
        }

        func block(_ inputs: [MLXArray]) -> [MLXArray] {
            var value = inputs[0]
            var cacheK = inputs[1], cacheV = inputs[2]
            let e = split(w["modulation"]!.expandedDimensions(axis: 0) + w["time_e0"]!, parts: 6, axis: 2)
                .map { $0.squeezed(axis: 2) }
            var hidden = layerNorm(value, meta.eps).asType(.float32) * (1 + e[1]) + e[0]
            let qkv = split(linear(hidden, "qkv_weight", "qkv_bias"), parts: 3, axis: -1)
            let query = ropeReal(rmsNorm(qkv[0], "norm_q", meta.attn_eps).reshaped([1, t, h, d]))
            let key = ropeReal(rmsNorm(qkv[1], "norm_k", meta.attn_eps).reshaped([1, t, h, d]))
            cacheK[0..., (end - t) ..< end] = key
            cacheV[0..., (end - t) ..< end] = qkv[2].reshaped([1, t, h, d])
            let attended = attention(query, cacheK, cacheV)
            value = value + linear(attended.reshaped([1, t, dim]), "o_weight", "o_bias") * e[2]
            let cam = w["cam"]!
            value = (1.0 + cam[0]) * value + cam[1]
            let q2 = rmsNorm(
                linear(layerNorm(value, meta.norm3_eps, weight: "norm3_weight", bias: "norm3_bias"),
                       "cross_q_weight", "cross_q_bias"),
                "cross_norm_q", meta.cross_eps
            ).reshaped([1, t, h, d])
            value = value + linear(attention(q2, w["text_k"]!, w["text_v"]!).reshaped([1, t, dim]),
                                   "cross_o_weight", "cross_o_bias")
            hidden = layerNorm(value, meta.eps).asType(.float32) * (1 + e[4]) + e[3]
            let up = linear(hidden, "ffn_up_weight", "ffn_up_bias").asType(.float32)
            let gelu = (0.5 * up * (1 + tanh(0.7978845608028654 * (up + 0.044715 * up * up * up)))).asType(.bfloat16)
            return [value + linear(gelu, "ffn_down_weight", "ffn_down_bias") * e[5], cacheK, cacheV]
        }

        func freshInputs() -> [MLXArray] {
            let inputs = [w["value"]!, w["cache_k"]! + 0, w["cache_v"]! + 0]  // copies: the block writes the cache
            eval(inputs)
            return inputs
        }
        func compare(_ label: String, _ ours: MLXArray, _ reference: MLXArray) {
            let a = ours.asType(.float32), b = reference.asType(.float32)
            let relative = (sqrt(sum((a - b) * (a - b))) / sqrt(sum(b * b))).item(Float.self)
            let maxAbs = abs(a - b).max().item(Float.self)
            log(String(format: "  %@ relative L2 %.2e  max abs %.3g  (|ref| max %.3g)", label, relative, maxAbs,
                       abs(b).max().item(Float.self)))
        }
        func time(_ fn: ([MLXArray]) -> [MLXArray]) -> Double {
            for _ in 0 ..< 3 { eval(fn(freshInputs())) }
            var samples: [Double] = []
            for _ in 0 ..< 10 {
                let inputs = freshInputs()
                let start = DispatchTime.now().uptimeNanoseconds
                eval(fn(inputs))
                samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            return samples.sorted()[samples.count / 2]
        }

        let eager = block(freshInputs())
        eval(eager)
        log("MLX-Swift on this device vs PyTorch MPS on the Mac:")
        compare("block output", eager[0], w["output"]!)
        compare("cache keys  ", eager[1], w["cache_k_after"]!)
        let compiled = compile(block)
        let compiledOut = compiled(freshInputs())
        eval(compiledOut)
        compare("compiled out", compiledOut[0], w["output"]!)
        let eagerMs = time(block), compiledMs = time(compiled)
        let torchMs = meta.torch_block_ms
        log(String(format: "time per block: Mac PyTorch %.1f ms | this device eager %.1f ms | compiled %.1f ms",
                   torchMs, eagerMs, compiledMs))
        log(String(format: "-> DiT per chunk (30 blocks x 4 passes): Mac PyTorch %.1f s, this device %.1f s (compiled)",
                   torchMs * 0.12, compiledMs * 0.12))
        log("DONE")
    }
}
