import Foundation
import MLX

/// Qwen3.5's linear attention (a Gated DeltaNet), as mlx-lm's qwen3_5.GatedDeltaNet computes
/// it, and the recurrent state it keeps instead of keys and values.
extension LlamaShard {
    /// mlx-lm's ArraysCache(size: 2) for one linear attention layer: the last (kernel - 1)
    /// inputs of its convolution and its state, for one sequence or a batch (a row each).
    /// Unlike keys and values, the state can't go back token by token: a node goes back to
    /// copies of it.
    struct RecurrentCache {
        var conv: MLXArray?
        var state: MLXArray?
        var leftPadding: [Int]?
        var lengths: [Int]?

        var isEmpty: Bool { conv == nil }

        /// ArraysCache.batch_size
        var rows: Int {
            conv?.dim(0) ?? state?.dim(0) ?? leftPadding?.count ?? lengths?.count ?? 1
        }

        mutating func advance(_ count: Int) {
            lengths = lengths?.map { $0 - count }
            leftPadding = leftPadding?.map { $0 - count }
        }

        /// ArraysCache.make_mask: which of the next `count` tokens of each row are real
        func mask(_ count: Int) -> MLXArray? {
            let positions = MLXArray(Int32(0) ..< Int32(count)).expandedDimensions(axis: 0)
            var mask: MLXArray? = nil
            if let leftPadding {
                mask = positions .>= MLXArray(leftPadding.map(Int32.init)).expandedDimensions(axis: 1)
            }
            if let lengths {
                let inLength = positions .< MLXArray(lengths.map(Int32.init)).expandedDimensions(axis: 1)
                mask = mask.map { $0 .&& inLength } ?? inLength
            }
            return mask
        }

        /// ArraysCache.merge: the sequences' states as a batch's rows
        static func merge(_ caches: [RecurrentCache]) -> RecurrentCache {
            if caches.allSatisfy(\.isEmpty) {
                return RecurrentCache(leftPadding: Array(repeating: 0, count: caches.count))
            }
            func stacked(_ part: KeyPath<RecurrentCache, MLXArray?>) -> MLXArray? {
                guard let template = caches.lazy.compactMap({ $0[keyPath: part] }).first else { return nil }
                let empty = MLXArray.zeros([1] + Array(template.shape.dropFirst()), dtype: template.dtype)
                return concatenated(caches.map { $0[keyPath: part] ?? empty }, axis: 0)
            }
            return RecurrentCache(conv: stacked(\.conv), state: stacked(\.state))
        }

        /// ArraysCache.extend: the other batch's rows after these
        mutating func extend(_ other: RecurrentCache) {
            let (ownRows, otherRows) = (rows, other.rows)
            func joined(_ own: MLXArray?, _ theirs: MLXArray?) -> MLXArray? {
                guard let template = theirs ?? own else { return nil }
                let tail = Array(template.shape.dropFirst())
                return concatenated([
                    own ?? MLXArray.zeros([ownRows] + tail, dtype: template.dtype),
                    theirs ?? MLXArray.zeros([otherRows] + tail, dtype: template.dtype),
                ], axis: 0)
            }
            func joined(_ own: [Int]?, _ theirs: [Int]?) -> [Int]? {
                if own == nil && theirs == nil { return nil }
                return (own ?? Array(repeating: 0, count: ownRows))
                    + (theirs ?? Array(repeating: 0, count: otherRows))
            }
            conv = joined(conv, other.conv)
            state = joined(state, other.state)
            leftPadding = joined(leftPadding, other.leftPadding)
            lengths = joined(lengths, other.lengths)
        }

        /// ArraysCache.filter: keep these rows
        mutating func filter(_ rows: [Int]) {
            let index = MLXArray(rows.map(Int32.init))
            conv = conv.map { $0[index] }
            state = state.map { $0[index] }
            leftPadding = leftPadding.map { padding in rows.map { padding[$0] } }
            lengths = lengths.map { lengths in rows.map { lengths[$0] } }
        }

        /// ArraysCache.extract: one row, as one sequence's cache
        func extract(_ row: Int) -> RecurrentCache {
            RecurrentCache(conv: conv?[row ..< row + 1], state: state?[row ..< row + 1])
        }

        /// ArraysCache.finalize
        mutating func finalize() {
            lengths = nil
            leftPadding = nil
        }

        var arrays: [MLXArray] { [conv, state].compactMap { $0 } }
    }

    static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    static func softplus(_ x: MLXArray) -> MLXArray { logAddExp(x, MLXArray(0).asType(x.dtype)) }

    /// A linear attention layer's mixing of its normed input, given its cache
    static func linearAttention(_ x: MLXArray, layer: Layer, cache: inout RecurrentCache) -> MLXArray {
        let config = layer.config
        let (batch, length) = (x.dim(0), x.dim(1))
        let (keyHeads, valueHeads) = (config.linearKeyHeads, config.linearValueHeads)
        let (keyDim, valueDim) = (config.linearKeyDim, config.linearValueDim)
        let keyWidth = keyHeads * keyDim
        let convWidth = 2 * keyWidth + valueHeads * valueDim
        let kept = config.convKernel - 1
        let mask = cache.mask(length)

        var qkv = layer.linear(x, "linear_attn.in_proj_qkv")
        let z = layer.linear(x, "linear_attn.in_proj_z").reshaped(batch, length, valueHeads, valueDim)
        let b = layer.linear(x, "linear_attn.in_proj_b")
        let a = layer.linear(x, "linear_attn.in_proj_a")

        let convState = cache.conv ?? MLXArray.zeros([batch, kept, convWidth], dtype: x.dtype)
        if let mask {
            qkv = which(mask.expandedDimensions(axis: -1), qkv, MLXArray(0).asType(qkv.dtype))
        }
        let convInput = concatenated([convState, qkv], axis: 1)
        if let lengths = cache.lengths {
            let ends = clip(MLXArray(lengths.map(Int32.init)), min: 0, max: length)
            let positions = (ends.expandedDimensions(axis: 1) + MLXArray(Int32(0) ..< Int32(kept)))
                .expandedDimensions(axis: -1)
            cache.conv = takeAlong(convInput, positions, axis: 1)
        } else {
            cache.conv = contiguous(convInput[0..., (-kept)..., 0...])
        }
        let convOut = silu(conv1d(convInput, layer.weight("linear_attn.conv1d.weight"), groups: convWidth))

        let parts = split(convOut, indices: [keyWidth, 2 * keyWidth], axis: -1)
        var q = parts[0].reshaped(batch, length, keyHeads, keyDim)
        var k = parts[1].reshaped(batch, length, keyHeads, keyDim)
        let v = parts[2].reshaped(batch, length, valueHeads, valueDim)

        // gated_delta.normalize_qk: L2-normalized, with the Dk ** -0.5 readout scale in q
        let inverseScale = 1 / Float(keyDim).squareRoot()
        let ones = MLXArray.ones([keyDim], dtype: q.dtype)
        let normEpsilon = 1e-6 * inverseScale * inverseScale
        q = MLXFast.rmsNorm(q, weight: ones, eps: normEpsilon) * (inverseScale * inverseScale)
        k = MLXFast.rmsNorm(k, weight: ones, eps: normEpsilon) * inverseScale

        let beta = sigmoid(b)
        let g = exp(
            -exp(layer.weight("linear_attn.A_log").asType(.float32))
                * softplus(a + layer.weight("linear_attn.dt_bias")))
        let state = cache.state ?? MLXArray.zeros([batch, valueHeads, valueDim, keyDim], dtype: .float32)
        let (out, newState) = gatedDelta(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        cache.state = newState
        cache.advance(length)

        // Qwen3NextRMSNormGated with precise_swiglu: normed, gated by silu(z) in float32
        let normed = MLXFast.rmsNorm(out, weight: layer.weight("linear_attn.norm.weight"), eps: config.normEpsilon)
        let gated = (silu(z.asType(.float32)) * normed.asType(.float32)).asType(out.dtype)
        return layer.linear(gated.reshaped(batch, length, -1), "linear_attn.out_proj")
    }

    /// gated_delta.gated_delta_kernel: the delta rule's recurrence over each row's tokens, one
    /// SIMD group per value row. mlx-lm runs a packed kernel when nothing is masked, which it
    /// documents as bitwise identical to this one on Apple GPUs.
    static func gatedDelta(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (batch, length, keyHeads, keyDim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (valueHeads, valueDim) = (v.dim(2), v.dim(3))
        var inputs: [any ScalarOrArray] = [q, k, v, g, beta, state, length]
        if let mask { inputs.append(mask) }
        let outputs = (mask == nil ? gatedDeltaKernel : gatedDeltaMaskedKernel)(
            inputs,
            template: [
                ("InT", q.dtype), ("StT", state.dtype), ("Dk", keyDim), ("Dv", valueDim),
                ("Hk", keyHeads), ("Hv", valueHeads),
            ],
            grid: (32, valueDim, batch * valueHeads),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, length, valueHeads, valueDim], state.shape],
            outputDTypes: [q.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }

    static let gatedDeltaKernel = makeGatedDeltaKernel(masked: false)
    static let gatedDeltaMaskedKernel = makeGatedDeltaKernel(masked: true)

    /// gated_delta._make_gated_delta_kernel(has_mask, vectorized=False), verbatim
    static func makeGatedDeltaKernel(masked: Bool) -> MLXFast.MLXFastKernel {
        let maskSource = masked ? "mask[b_idx * T + t]" : "true"
        let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
            """
        return MLXFast.metalKernel(
            name: masked ? "gated_delta_step_mask" : "gated_delta_step",
            inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"] + (masked ? ["mask"] : []),
            outputNames: ["y", "state_out"],
            source: source)
    }
}
