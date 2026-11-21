import Foundation
import MLX

/// Attention over two parts of the keys merged by their log-sum-exp (SplitStage's overlap) against one
/// attention call, on random data at the split's shapes: eager, compiled, and compiled with the first
/// part async-evaluated first, as SplitStage runs it. SPIKE_MODE=lsecheck
enum LSECheck {
    static func run(log: @escaping (String) -> Void) {
        let heads = 12, d = 128, scale = 1 / Float(d).squareRoot()
        func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                scale: scale, mask: nil
            ).transposed(0, 2, 1, 3)
        }
        func attentionLSE(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> [MLXArray] {
            let (out, lse) = MLXFast.scaledDotProductAttentionWithLSE(
                queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                scale: scale)
            return [out.transposed(0, 2, 1, 3), lse.transposed(0, 2, 1, 3)]
        }
        func mergeParts(_ a: [MLXArray]) -> [MLXArray] {
            let lse = logAddExp(a[1], a[3])
            return [(a[0].asType(.float32) * exp(a[1] - lse) + a[2].asType(.float32) * exp(a[3] - lse)).asType(a[0].dtype)]
        }
        let attendLSE = compile { a in attentionLSE(a[0], a[1], a[2]) }
        let merge = compile(mergeParts)
        func report(_ label: String, _ x: MLXArray, _ ref: MLXArray) {
            let a = x.asType(.float32), b = ref.asType(.float32)
            let nans = isNaN(a).sum().item(Int.self)
            let rel = (sqrt(sum((a - b) * (a - b))) / sqrt(sum(b * b))).item(Float.self)
            log(String(format: "    %@: NaN %d, relative L2 %.2e", label, nans, rel))
        }
        for (qL, total, rows) in [(2960, 12064, 6032), (2064, 27144, 21112), (6032, 12064, 6032), (2944, 27144, 21112)] {
            let q = (MLXRandom.normal([1, qL, heads, d]) * 1.5).asType(.bfloat16)
            let k = (MLXRandom.normal([1, total, heads, d]) * 1.5).asType(.bfloat16)
            let v = MLXRandom.normal([1, total, heads, d]).asType(.bfloat16)
            eval(q, k, v)
            log("\(qL) queries, \(rows) older + \(total - rows) chunk rows:")
            let ref = attention(q, k, v)
            let partA = attentionLSE(q, k[0..., ..<rows], v[0..., ..<rows])
            eval(partA)
            log(String(format: "    lse A: NaN %d, range %.2f..%.2f, shape %@", isNaN(partA[1]).sum().item(Int.self),
                       partA[1].min().item(Float.self), partA[1].max().item(Float.self), "\(partA[1].shape)"))
            report("eager   ", mergeParts(partA + attentionLSE(q, k[0..., rows...], v[0..., rows...]))[0], ref)
            report("compiled", merge(attendLSE([q, k[0..., ..<rows], v[0..., ..<rows]])
                                     + attendLSE([q, k[0..., rows...], v[0..., rows...]]))[0], ref)
            let older = attendLSE([q, k[0..., ..<rows], v[0..., ..<rows]])
            asyncEval(older)
            report("async   ", merge(older + attendLSE([q, k[0..., rows...], v[0..., rows...]]))[0], ref)
        }
        log("DONE")
    }
}
