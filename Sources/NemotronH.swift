import Foundation
import MLX

/// Nemotron-H's blocks, as exo runs them (its models/nemotron_h/model.py: mlx-lm's nemotron_h,
/// with the Mamba-2 mixer between its projections, and the router, in float32). Each block is
/// a norm and one mixer: Mamba-2 ("M", a recurrent state), attention without RoPE ("*"), an MLP
/// ("-") or a mixture of experts ("E"); only the first two have a cache.
extension LlamaShard {
    enum NemotronBlock {
        case mamba, attention, mlp, experts
    }

    /// x + mixer(norm(x)), for a block given its mixer
    static func nemotronBlock(_ x: MLXArray, layer: Layer, mix: (MLXArray) -> MLXArray) -> MLXArray {
        x + mix(MLXFast.rmsNorm(x, weight: layer.weight("norm.weight"), eps: layer.config.normEpsilon))
    }

    /// relu2: max(x, 0) squared
    static func relu2(_ x: MLXArray) -> MLXArray {
        square(maximum(x, MLXArray(0).asType(x.dtype)))
    }

    /// NemotronHMLP
    static func nemotronMLP(_ x: MLXArray, layer: Layer, prefix: String) -> MLXArray {
        layer.linear(relu2(layer.linear(x, prefix + ".up_proj")), prefix + ".down_proj")
    }

    /// NemotronHAttention: no RoPE, no norms of queries and keys
    static func nemotronAttention<Cache: LayerCache>(
        _ x: MLXArray, layer: Layer, cache: inout Cache
    ) -> MLXArray {
        let config = layer.config
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries = layer.linear(x, "mixer.q_proj")
            .reshaped(batch, length, config.heads, -1).transposed(0, 2, 1, 3)
        var keys = layer.linear(x, "mixer.k_proj")
            .reshaped(batch, length, config.kvHeads, -1).transposed(0, 2, 1, 3)
        var values = layer.linear(x, "mixer.v_proj")
            .reshaped(batch, length, config.kvHeads, -1).transposed(0, 2, 1, 3)
        let mask = cache.mask(length)
        (keys, values) = cache.update(keys: keys, values: values)
        // head_dim ** -0.5, as Python computes it and rounds it to float32
        let scale = Float(pow(Double(config.headDim), -0.5))
        let attention = attend(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return layer.linear(attention, "mixer.o_proj")
    }

    /// NemotronHMoE with exo's Float32RouterGate: each token's top k experts by the sigmoid of
    /// float32 router logits plus a correction bias, weighed by their uncorrected scores, plus a
    /// shared expert
    static func nemotronExperts(_ x: MLXArray, layer: Layer) -> MLXArray {
        let config = layer.config
        let k = config.expertsPerToken
        let logits = matmul(x.asType(.float32), layer.weight("mixer.gate.weight").T.asType(.float32))
        // group_expert_select (n_group 1: a stage refuses grouped routing at load)
        let original = sigmoid(logits)
        let corrected = original + layer.weight("mixer.gate.e_score_correction_bias")
        let experts = argPartition(-corrected, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(original, experts, axis: -1)
        if k > 1 && config.normalizeExpertScores {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        scores = scores * MLXArray(config.routedScalingFactor)

        // SwitchMLP: [B, L, 1, 1, D], one row per token, for each of its experts to project
        var rows = expandedDimensions(x, axes: [-2, -3])
        var rowExperts = experts
        let sorted = experts.size >= 64
        var inverseOrder: MLXArray? = nil
        if sorted {
            let flatExperts = experts.flattened()
            let order = argSort(flatExperts)
            inverseOrder = argSort(order)
            rows = rows.flattened(start: 0, end: -3)[order.floorDivide(k)]
            rowExperts = flatExperts[order]
        }
        var projected = layer.expertLinear(
            relu2(layer.expertLinear(rows, "mixer.switch_mlp.fc1", experts: rowExperts, sorted: sorted)),
            "mixer.switch_mlp.fc2", experts: rowExperts, sorted: sorted)
        if let inverseOrder {
            projected = unflatten(projected[inverseOrder], axis: 0, shape: experts.shape)
        }
        projected = projected.squeezed(axis: -2)
        let routed = (projected * expandedDimensions(scores, axis: -1)).sum(axis: -2).asType(projected.dtype)
        guard layer.weights["mixer.shared_experts.up_proj.weight"] != nil else { return routed }
        return routed + nemotronMLP(x, layer: layer, prefix: "mixer.shared_experts")
    }

    /// exo's Float32Mamba2Mixer: the causal conv, the scan and the gated norm in float32, the
    /// result rounded once for out_proj
    static func mamba2(_ x: MLXArray, layer: Layer, cache: inout RecurrentCache) -> MLXArray {
        let config = layer.config
        let (batch, length) = (x.dim(0), x.dim(1))
        let (heads, headDim) = (config.mambaHeads, config.mambaHeadDim)
        let (groups, stateSize) = (config.mambaGroups, config.ssmStateSize)
        let inner = heads * headDim
        let convDim = inner + 2 * groups * stateSize
        let kept = config.convKernel - 1
        let mask = cache.mask(length)

        let parts = split(layer.linear(x, "mixer.in_proj"), indices: [inner, inner + convDim], axis: -1)
        let gate = parts[0]
        var convInput = parts[1]
        let dt = parts[2]

        // _conv
        if let mask {
            convInput = which(mask.expandedDimensions(axis: -1), convInput, MLXArray(0).asType(convInput.dtype))
        }
        let convState = cache.conv ?? MLXArray.zeros([batch, kept, convDim], dtype: convInput.dtype)
        let padded = concatenated([convState, convInput], axis: 1)
        if let lengths = cache.lengths {
            let ends = clip(MLXArray(lengths.map(Int32.init)), min: 0, max: padded.dim(1) - kept)
            let positions = (ends.expandedDimensions(axis: 1) + MLXArray(Int32(0) ..< Int32(kept)))
                .expandedDimensions(axis: -1)
            cache.conv = takeAlong(padded, positions, axis: 1)
        } else {
            cache.conv = padded[0..., (-kept)..., 0...]
        }
        // Float32Conv1d, then SiLU
        var convOut = conv1d(
            padded.asType(.float32), layer.weight("mixer.conv1d.weight").asType(.float32), groups: convDim)
        if let bias = layer.weights["mixer.conv1d.bias"] {
            convOut = convOut + bias.asType(.float32)
        }
        convOut = silu(convOut)

        let pieces = split(convOut, indices: [inner, inner + groups * stateSize], axis: -1)
        let hidden = pieces[0].reshaped(batch, length, heads, headDim)
        let b = pieces[1].reshaped(batch, length, groups, stateSize)
        let c = pieces[2].reshaped(batch, length, groups, stateSize)
        let (y, state) = ssmUpdate(
            hidden, aLog: layer.weight("mixer.A_log"), b: b, c: c,
            d: layer.weight("mixer.D").asType(hidden.dtype), dt: dt, dtBias: layer.weight("mixer.dt_bias"),
            state: cache.state, limit: config.timeStepLimit, mask: mask)
        cache.state = state
        cache.advance(length)

        // MambaRMSNormGated, given a float32 gate: silu(gate) * y, normed per group
        var normed = silu(gate.asType(.float32)) * y.reshaped(batch, length, inner)
        normed = unflatten(normed, axis: -1, shape: [-1, inner / groups])
        normed = MLXFast.rmsNorm(
            normed, weight: MLXArray.ones([inner / groups], dtype: normed.dtype), eps: config.normEpsilon)
        normed = layer.weight("mixer.norm.weight") * normed.flattened(start: -2)
        return layer.linear(normed.asType(x.dtype), "mixer.out_proj")
    }

    /// ssm.compute_dt
    static func computeDt(_ dt: MLXArray, bias: MLXArray, limit: (Float, Float)) -> MLXArray {
        clip(softplus(dt.asType(.float32) + bias), min: limit.0, max: limit.1)
    }

    /// ssm.ssm_update: the kernel for one token given a state, else the chunked scan
    static func ssmUpdate(
        _ x: MLXArray, aLog: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, dt: MLXArray,
        dtBias: MLXArray, state: MLXArray?, limit: (Float, Float), mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        guard x.dim(1) == 1, let state else {
            return ssmChunked(x, aLog: aLog, b: b, c: c, d: d, dt: dt, dtBias: dtBias, state: state, limit: limit, mask: mask)
        }
        let (rows, heads, headDim) = (x.dim(0), x.dim(2), x.dim(3))
        let (groups, stateSize) = (b.dim(-2), b.dim(-1))
        let outputs = ssmKernel(
            [x, aLog, b, c, d, computeDt(dt, bias: dtBias, limit: limit), state],
            template: [
                ("T", x.dtype), ("U", state.dtype), ("Dh", headDim), ("Ds", stateSize), ("H", heads),
                ("G", heads / groups),
            ],
            grid: (32, headDim, heads * rows),
            threadGroup: (32, 8, 1),
            outputShapes: [[rows, 1, heads, headDim], state.shape],
            outputDTypes: [x.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }

    /// ssm.segsum
    static func segsum(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let length = x.dim(-1)
        var x = x
        var mask = mask
        if let given = mask {
            mask = expandedDimensions(given, axis: 1)
            x = x * mask!
        }
        x = repeated(expandedDimensions(x, axis: -1), count: length, axis: -1)
        x = tril(x, k: -1)
        var sums = cumsum(x, axis: -2)
        if let mask {
            sums = which(
                expandedDimensions(mask, axis: -2) .&& expandedDimensions(mask, axis: -1), sums,
                MLXArray(-Float.infinity).asType(sums.dtype))
        }
        return sums
    }

    /// ssm.ssm_attn: the SSD scan over chunks of 256 tokens
    static func ssmChunked(
        _ x: MLXArray, aLog: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray, dt: MLXArray,
        dtBias: MLXArray, state: MLXArray?, limit: (Float, Float), mask: MLXArray?, step: Int = 256
    ) -> (MLXArray, MLXArray) {
        let (rows, length, heads, headDim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (groups, stateSize) = (b.dim(2), b.dim(3))
        let dt = computeDt(dt, bias: dtBias, limit: limit)
        let repeats = heads / groups
        let a = -exp(aLog).asType(dt.dtype)
        var dtA = dt * a.reshaped(1, 1, -1)
        if let mask {
            dtA = dtA * mask.expandedDimensions(axis: -1)
        }
        let dtx = dt.reshaped(rows, length, heads, 1) * x

        func chunk(
            _ dtx: MLXArray, _ dtA: MLXArray, _ b: MLXArray, _ c: MLXArray, _ state: MLXArray?, _ mask: MLXArray?
        ) -> (MLXArray, MLXArray) {
            let size = dtx.dim(1)
            var b = b.transposed(0, 2, 3, 1)
            let cb = repeated(matmul(c.swappedAxes(1, 2), b), count: repeats, axis: 1)
            var decay = exp(segsum(dtA.swappedAxes(1, 2), mask: mask))
            let surrogate = tril(cb * decay, k: 0)
            var y = matmul(surrogate, dtx.swappedAxes(1, 2)).swappedAxes(1, 2)
            decay = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
            b = repeated(b, count: heads / groups, axis: 1).swappedAxes(2, 3)
            let weighed = (dtx * decay).swappedAxes(1, 2).swappedAxes(2, 3)
            var next = matmul(weighed, b)
            if let state {
                let decayed = exp(cumsum(dtA, axis: -2))
                next = next + decayed[0..., -1, 0..., .newAxis, .newAxis] * state
                let cc = c.reshaped(rows, size, groups, 1, stateSize, 1)
                let previous = matmul(state.reshaped(rows, 1, groups, repeats, headDim, stateSize), cc)
                    .squeezed(axis: -1).flattened(start: 2, end: 3)
                y = y + decayed.expandedDimensions(axis: -1) * previous
            }
            return (y.asType(x.dtype), next)
        }

        var state = state
        var ys: [MLXArray] = []
        for start in stride(from: 0, to: length, by: step) {
            let end = min(start + step, length)
            let (y, next) = chunk(
                dtx[0..., start ..< end], dtA[0..., start ..< end], b[0..., start ..< end],
                c[0..., start ..< end], state, mask.map { $0[.ellipsis, start ..< end] })
            ys.append(y)
            state = next
        }
        let y = concatenated(ys, axis: 1) + x * d.reshaped(1, 1, heads, 1)
        return (y, state!)
    }

    /// ssm.make_ssm_kernel, verbatim
    static let ssmKernel = MLXFast.metalKernel(
        name: "ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = Ds / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            // C and B have shape [batch, group, state_dim]
            // C and B need to be offset by group size
            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * i_state[idx] + dB_by_x;
                o_state[idx] = static_cast<U>(state);
                acc += state * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
            }
            """)
}
