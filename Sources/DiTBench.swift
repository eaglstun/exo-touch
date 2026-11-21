import Foundation
import MLX

/// How fast this device runs FastFrames-LingBot's DiT (Wan 2.1 1.3B blocks), at the shapes one
/// chunk of the camera-control demo uses: 4 latent frames of queries against a rolling KV cache of
/// 18 frames, BF16 operands. The Mac runs the same shapes with fastframes' scripts/bench_dit_block.py;
/// the two decide whether splitting a chunk's tokens between the Mac and this iPad can pay off.
///
///     SPIKE_MODE=dit (devicectl --environment-variables), or mac/run_dit_bench.sh
enum DiTBench {
    static let dim = 1536, heads = 12, headDim = 128, ffn = 8960, textTokens = 512
    static let blocks = 30, passes = 4  // DiT blocks; denoising passes per chunk
    static let chunkFrames = 4, cacheFrames = 18

    /// The demo's frame sizes and their DiT tokens per latent frame (16x16 pixels per token)
    static let sizes: [(name: String, frameTokens: Int)] = [
        ("832x464", 52 * 29), ("576x320", 36 * 20), ("416x240", 26 * 15),
    ]

    static func run(log: @escaping (String) -> Void) {
        log("device: \(Device.defaultDevice())  DiT block shapes, bf16; median of 10 after 3 warm-ups")
        for size in sizes {
            bench(size: size.name, queries: chunkFrames * size.frameTokens,
                  keys: cacheFrames * size.frameTokens, log: log)
        }
        log("DONE")
    }

    private static func bench(size: String, queries q: Int, keys k: Int, log: (String) -> Void) {
        func weight(_ rows: Int, _ cols: Int) -> MLXArray {
            (MLXRandom.normal([rows, cols]) * 0.02).asType(.bfloat16)
        }
        let x = MLXRandom.normal([q, dim]).asType(.bfloat16)
        let wqkv = weight(dim, 3 * dim), wo = weight(dim, dim), wcam = weight(dim, 2 * dim)
        let wq2 = weight(dim, dim), wo2 = weight(dim, dim), wup = weight(dim, ffn), wdown = weight(ffn, dim)
        let hidden = MLXRandom.normal([q, ffn]).asType(.bfloat16)
        let query = MLXRandom.normal([1, heads, q, headDim]).asType(.bfloat16)
        let cacheK = MLXRandom.normal([1, heads, k, headDim]).asType(.bfloat16)
        let cacheV = MLXRandom.normal([1, heads, k, headDim]).asType(.bfloat16)
        let textK = MLXRandom.normal([1, heads, textTokens, headDim]).asType(.bfloat16)
        let textV = MLXRandom.normal([1, heads, textTokens, headDim]).asType(.bfloat16)
        let residual = MLXRandom.normal([q, dim])  // float32, like the DiT's residual stream
        eval(x, wqkv, wo, wcam, wq2, wo2, wup, wdown, hidden, query, cacheK, cacheV, textK, textV, residual)
        let scale = 1 / Float(headDim).squareRoot()

        func gemm(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let ms = time { matmul(a, b) }
            let flops = 2.0 * Double(a.dim(0)) * Double(a.dim(1)) * Double(b.dim(1))
            log(String(format: "%@ %-22@ %8.2f ms %6.2f TFLOPS", size, name, ms, flops / ms / 1e9))
        }
        gemm("gemm qkv", x, wqkv)
        gemm("gemm out", x, wo)
        gemm("gemm camera", x, wcam)
        gemm("gemm ffn up", x, wup)
        gemm("gemm ffn down", hidden, wdown)

        let selfMs = time {
            MLXFast.scaledDotProductAttention(queries: query, keys: cacheK, values: cacheV, scale: scale, mask: nil)
        }
        let selfFlops = 4.0 * Double(heads) * Double(q) * Double(k) * Double(headDim)
        log(String(format: "%@ %-22@ %8.2f ms %6.2f TFLOPS", size, "self-attention \(q)x\(k)", selfMs,
                   selfFlops / selfMs / 1e9))
        let crossMs = time {
            MLXFast.scaledDotProductAttention(queries: query, keys: textK, values: textV, scale: scale, mask: nil)
        }
        log(String(format: "%@ %-22@ %8.2f ms", size, "cross-attention", crossMs))

        // One whole block, as the DiT runs it: FP32 residual stream, BF16 matmuls
        func heads4(_ t: MLXArray) -> MLXArray { t.reshaped([1, q, heads, headDim]).transposed(0, 2, 1, 3) }
        func merge(_ t: MLXArray) -> MLXArray { t.transposed(0, 2, 1, 3).reshaped([q, dim]) }
        let blockMs = time {
            var value = residual
            let normed = MLXFast.layerNorm(value, eps: 1e-6).asType(.bfloat16)
            let qkv = matmul(normed, wqkv)
            let parts = split(qkv, parts: 3, axis: -1)
            let attended = MLXFast.scaledDotProductAttention(
                queries: heads4(parts[0]),
                keys: concatenated([cacheK[0..., 0..., (q)...], heads4(parts[1])], axis: 2),
                values: concatenated([cacheV[0..., 0..., (q)...], heads4(parts[2])], axis: 2),
                scale: scale, mask: nil)
            value = value + matmul(merge(attended), wo).asType(.float32)
            let camera = matmul(value.asType(.bfloat16), wcam)
            let cameraParts = split(camera, parts: 2, axis: -1)
            value = value * (1 + cameraParts[0].asType(.float32)) + cameraParts[1].asType(.float32)
            let crossQ = heads4(matmul(MLXFast.layerNorm(value, eps: 1e-6).asType(.bfloat16), wq2))
            let crossed = MLXFast.scaledDotProductAttention(queries: crossQ, keys: textK, values: textV, scale: scale,
                                                            mask: nil)
            value = value + matmul(merge(crossed), wo2).asType(.float32)
            let up = matmul(MLXFast.layerNorm(value, eps: 1e-6).asType(.bfloat16), wup)
            let activated = up * sigmoid(1.702 * up)  // gelu (tanh-free approximation, same cost class)
            return value + matmul(activated, wdown).asType(.float32)
        }
        let chunkSeconds = blockMs * Double(blocks * passes) / 1000
        log(String(format: "%@ %-22@ %8.2f ms  -> DiT per chunk (30 blocks x 4 passes) %.2f s", size,
                   "whole block", blockMs, chunkSeconds))
    }

    /// Median milliseconds of `body` over 10 timed runs after 3 warm-ups, each evaluated to completion
    private static func time(_ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        var samples: [Double] = []
        for _ in 0 ..< 10 {
            let start = DispatchTime.now().uptimeNanoseconds
            eval(body())
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        return samples.sorted()[samples.count / 2]
    }
}
