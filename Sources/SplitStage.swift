import Foundation
import MLX

/// This iPad's share of FastFrames-LingBot's DiT, in step with a Mac (fastframes' apps/lingbot/ipad.py,
/// IPadAssist). Every denoising pass's chunk tokens are split: the Mac keeps [0, start), this device
/// runs [start, tokens) through all 30 blocks. In each block both sides compute their tokens' keys
/// and values, swap the canonical (pre-RoPE) ones, rotate the other side's with the chunk's RoPE
/// table, write both into their own KV cache and attend over all of it; the rest of a block is per
/// token. The block math is DiTBlockCheck's (checked against PyTorch).
///
/// A session mirrors the Mac's cache: weights once (setup, layer), the prompt's text K/V (context),
/// zeroed caches (reset), the Mac's eviction copies (evict), each chunk's RoPE table and this
/// side's camera tokens, from which it computes its camera rows (chunk), and the refined rows the
/// Mac commits after a chunk (rows).
///
/// Frames both ways: little-endian u32 header length, a JSON header, then the arrays' bytes in the
/// header's order: {"op", "layer", "meta", "arrays": [{"name", "shape", "dtype"}]}.
///   Mac -> iPad: setup, layer x30, context x30, reset, evict, chunk, pass, kv, rows x30, bye
///   iPad -> Mac: ready, kv, result, error
enum SplitStage {
    static let defaultPort: UInt16 = 5570

    struct Message {
        var op: String
        var layer = 0
        var meta: [String: Any] = [:]
        var arrays: [String: MLXArray] = [:]
    }

    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    /// Frames read on their own thread, so a pass can keep the GPU busy while it waits for one
    final class Inbox: @unchecked Sendable {
        private let condition = NSCondition()
        private var messages: [Message] = []
        private var closed = false

        func push(_ message: Message?) {
            condition.lock()
            if let message { messages.append(message) } else { closed = true }
            condition.signal()
            condition.unlock()
        }

        /// The next message; `idle` runs between checks while none has arrived
        func pop(idle: (() -> Void)? = nil) throws -> Message {
            while true {
                condition.lock()
                if !messages.isEmpty {
                    let message = messages.removeFirst()
                    condition.unlock()
                    return message
                }
                if closed {
                    condition.unlock()
                    throw RemoteStageServer.ConnectionClosed()
                }
                if let idle {
                    condition.unlock()
                    idle()
                } else {
                    condition.wait()
                    condition.unlock()
                }
            }
        }
    }

    static func serve(port: UInt16, log: @escaping @Sendable (String) -> Void) throws {
        let listener = socket(AF_INET, SOCK_STREAM, 0)
        var yes: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = INADDR_ANY
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(listener, 1) == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        log("split stage listening on port \(port)")
        while true {
            let connection = accept(listener, nil, nil)
            guard connection >= 0 else { continue }
            setsockopt(connection, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            do {
                try Session(connection: connection, log: log).run()
            } catch is RemoteStageServer.ConnectionClosed {
                log("Mac disconnected")
            } catch {
                log("FAILED: \(error)")
            }
            close(connection)
            Memory.clearCache()
        }
    }

    // MARK: frames

    static func itemSize(_ dtype: DType) -> Int { dtype == .float32 ? 4 : 2 }

    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .float32: "float32"
        case .float16: "float16"
        default: "bfloat16"
        }
    }

    static func read(_ connection: Int32) throws -> Message {
        let length = try RemoteStageServer.readExactly(connection, 4).withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
        }
        let header = try JSONSerialization.jsonObject(
            with: try RemoteStageServer.readExactly(connection, Int(length))) as! [String: Any]
        var message = Message(op: header["op"] as! String, layer: header["layer"] as? Int ?? 0,
                              meta: header["meta"] as? [String: Any] ?? [:])
        for spec in header["arrays"] as? [[String: Any]] ?? [] {
            let shape = spec["shape"] as! [Int]
            let dtype = RemoteStageServer.dtype(named: spec["dtype"] as! String)!
            message.arrays[spec["name"] as! String] = MLXArray(
                try RemoteStageServer.readExactly(connection, shape.reduce(1, *) * itemSize(dtype)), shape, dtype: dtype)
        }
        return message
    }

    /// Header and bytes of `arrays` (evaluated): built on the compute thread, written by the sender
    static func encode(op: String, layer: Int = 0, meta: [String: Any] = [:], arrays: [(String, MLXArray)]) -> Data {
        let specs = arrays.map { ["name": $0.0, "shape": $0.1.shape, "dtype": dtypeName($0.1.dtype)] as [String: Any] }
        let header = try! JSONSerialization.data(
            withJSONObject: ["op": op, "layer": layer, "meta": meta, "arrays": specs] as [String: Any])
        var length = UInt32(header.count).littleEndian
        var data = Data(bytes: &length, count: 4)
        data.append(header)
        for (_, array) in arrays { data.append(array.asData(access: .copy).data) }
        return data
    }

    // MARK: a session

    final class Session {
        let connection: Int32
        let log: @Sendable (String) -> Void
        let inbox = Inbox()
        let sender = DispatchQueue(label: "split-sender")

        var tokens = 0, start = 0, heads = 0, layerCount = 0, cacheRows = 0
        var eps: Float = 1e-6, attnEps: Float = 1e-6, crossEps: Float = 1e-6, norm3Eps: Float = 1e-6
        var cameraWeights: [String: MLXArray] = [:]
        var weights: [[String: MLXArray]] = []
        var cacheK: [MLXArray] = [], cacheV: [MLXArray] = []
        var textK: [MLXArray] = [], textV: [MLXArray] = []
        var camera: [MLXArray] = []  // this side's camera scale/shift rows of every block, [2, 1, share, dim]
        var rope = MLXArray(0)  // the chunk's RoPE table of every token, [tokens, 1, D/2, 2]
        var passes = 0

        var share: Int { tokens - start }

        init(connection: Int32, log: @escaping @Sendable (String) -> Void) {
            self.connection = connection
            self.log = log
        }

        func send(_ data: Data) {
            let connection = connection
            sender.async { try? RemoteStageServer.writeAll(connection, data) }
        }

        func run() throws {
            let inbox = inbox, connection = connection
            Thread {
                while true {
                    do { inbox.push(try read(connection)) } catch { inbox.push(nil); return }
                }
            }.start()
            do {
                while true {
                    let message = try inbox.pop()
                    if message.op == "bye" { return }
                    try handle(message)
                }
            } catch let error as Failure {
                send(encode(op: "error", meta: ["message": error.description], arrays: []))
                throw error
            }
        }

        func handle(_ message: Message) throws {
            switch message.op {
            case "setup":
                let meta = message.meta
                tokens = meta["tokens"] as! Int
                start = meta["start"] as! Int
                heads = meta["heads"] as! Int
                layerCount = meta["layers"] as! Int
                cacheRows = meta["cache_rows"] as! Int
                eps = Float(meta["eps"] as! Double)
                attnEps = Float(meta["attn_eps"] as! Double)
                crossEps = Float(meta["cross_eps"] as! Double)
                norm3Eps = Float(meta["norm3_eps"] as! Double)
                cameraWeights = message.arrays
                weights = []
                let started = Date()
                for _ in 0 ..< layerCount {
                    let layer = try inbox.pop()
                    guard layer.op == "layer" else { throw Failure(description: "expected layer, got \(layer.op)") }
                    eval(Array(layer.arrays.values))
                    weights.append(layer.arrays)
                }
                let dim = weights[0]["o_weight"]!.dim(0), headDim = dim / heads
                cacheK = (0 ..< layerCount).map { _ in MLXArray.zeros([1, cacheRows, heads, headDim], dtype: .bfloat16) }
                cacheV = (0 ..< layerCount).map { _ in MLXArray.zeros([1, cacheRows, heads, headDim], dtype: .bfloat16) }
                eval(cacheK + cacheV)
                textK = []
                textV = []
                compileBlocks(dim: dim)
                log(String(format: "setup: tokens %d..<%d of %d, %d layers, %d cache rows in %.1f s; footprint %.1f GB",
                           start, tokens, tokens, layerCount, cacheRows, Date().timeIntervalSince(started),
                           Double(RemoteStageServer.footprint()) / 1e9))
                send(encode(op: "ready", meta: ["footprint": RemoteStageServer.footprint()], arrays: []))
            case "context":
                if message.layer == 0 {
                    textK = []
                    textV = []
                }
                textK.append(message.arrays["text_k"]!)
                textV.append(message.arrays["text_v"]!)
                eval(textK.last!, textV.last!)
            case "reset":
                for i in 0 ..< layerCount {
                    cacheK[i] = MLXArray.zeros(cacheK[i].shape, dtype: .bfloat16)
                    cacheV[i] = MLXArray.zeros(cacheV[i].shape, dtype: .bfloat16)
                }
                eval(cacheK + cacheV)
            case "evict":
                // FastFramesLingBot._evict: per layer, blocks of `shift` rows moving left
                let sink = message.meta["sink"] as! Int, shift = message.meta["shift"] as! Int
                let rolled = message.meta["rolled"] as! Int
                for i in 0 ..< layerCount {
                    for from in stride(from: sink, to: sink + rolled, by: shift) {
                        let to = min(from + shift, sink + rolled)
                        cacheK[i][0..., from ..< to] = cacheK[i][0..., (from + shift) ..< (to + shift)]
                        cacheV[i][0..., from ..< to] = cacheV[i][0..., (from + shift) ..< (to + shift)]
                    }
                }
                eval(cacheK + cacheV)
            case "chunk":
                // The Mac may move the split between chunks (it balances the two sides' speeds)
                if let newStart = message.meta["start"] as? Int { start = newStart }
                rope = message.arrays["rope"]!
                camera = cameraRows(message.arrays["camera"]!)
                eval([rope] + camera)
            case "pass":
                try pass(message)
            case "rows":
                let end = message.meta["end"] as! Int
                cacheK[message.layer][0..., (end - tokens) ..< end] = message.arrays["k"]!
                cacheV[message.layer][0..., (end - tokens) ..< end] = message.arrays["v"]!
                eval(cacheK[message.layer], cacheV[message.layer])
            default:
                throw Failure(description: "unexpected \(message.op)")
            }
        }

        // MARK: the block

        func linear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
            addMM(bias, x.asType(.bfloat16), weight.T)
        }

        /// x * sigmoid(x) in FP32, rounded to x's dtype (PyTorch's F.silu on BF16)
        func silu(_ x: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            return (xf * sigmoid(xf)).asType(x.dtype)
        }

        /// LingBotDiT._prepare_camera for this side's tokens: [1, share, 1536] camera tokens -> every
        /// block's [2, 1, share, dim] scale and shift
        func cameraRows(_ tokens: MLXArray) -> [MLXArray] {
            let c = cameraWeights
            var hidden = linear(tokens, c["cam_patch_weight"]!, c["cam_patch_bias"]!)
            hidden = hidden + linear(silu(linear(hidden, c["c2ws1_weight"]!, c["c2ws1_bias"]!)),
                                     c["c2ws2_weight"]!, c["c2ws2_bias"]!)
            return weights.map { w in
                let injected = linear(silu(linear(hidden, w["inj1_weight"]!, w["inj1_bias"]!)),
                                      w["inj2_weight"]!, w["inj2_bias"]!) + hidden
                let both = split(linear(injected, w["cam_weight"]!, w["cam_bias"]!), parts: 2, axis: -1)
                return stacked(both, axis: 0)
            }
        }

        var pre: (([MLXArray]) -> [MLXArray])!
        var post: (([MLXArray]) -> [MLXArray])!
        var rotate: (([MLXArray]) -> [MLXArray])!

        func compileBlocks(dim: Int) {
            let heads = heads, d = dim / heads, eps = eps, attnEps = attnEps
            let crossEps = crossEps, norm3Eps = norm3Eps, scale = 1 / Float(d).squareRoot()
            func linear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
                addMM(bias, x.asType(.bfloat16), weight.T)
            }
            func layerNorm(_ x: MLXArray, _ eps: Float, _ weight: MLXArray? = nil, _ bias: MLXArray? = nil) -> MLXArray {
                MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32), bias: bias?.asType(.float32),
                                  eps: eps).asType(x.dtype)
            }
            func rmsNorm(_ x: MLXArray, _ weight: MLXArray, _ eps: Float) -> MLXArray {
                let xf = x.asType(.float32)
                return (xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)).asType(x.dtype) * weight
            }
            func ropeReal(_ x: MLXArray, _ table: MLXArray) -> MLXArray {
                let shape = x.shape
                let pairs = x.asType(.float32).reshaped(shape.dropLast() + [d / 2, 2])
                let xr = pairs[.ellipsis, 0], xi = pairs[.ellipsis, 1]
                let fr = table[.ellipsis, 0], fi = table[.ellipsis, 1]
                return stacked([xr * fr - xi * fi, xr * fi + xi * fr], axis: -1).reshaped(shape).asType(x.dtype)
            }
            func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
                MLXFast.scaledDotProductAttention(
                    queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                    scale: scale, mask: nil
                ).transposed(0, 2, 1, 3)
            }
            func modulation(_ m: MLXArray, _ timeE0: MLXArray) -> [MLXArray] {
                split(m.expandedDimensions(axis: 0) + timeE0, parts: 6, axis: 2).map { $0.squeezed(axis: 2) }
            }
            // [value, modulation, time_e0, qkv_weight, qkv_bias, norm_q, norm_k, rope rows]
            //   -> [RoPE'd query, canonical key, value rows]
            pre = compile { a in
                let tokens = a[0].dim(1)
                let e = modulation(a[1], a[2])
                let hidden = layerNorm(a[0], eps).asType(.float32) * (1 + e[1]) + e[0]
                let qkv = split(linear(hidden, a[3], a[4]), parts: 3, axis: -1)
                return [ropeReal(rmsNorm(qkv[0], a[5], attnEps).reshaped([1, tokens, heads, d]), a[7]),
                        rmsNorm(qkv[1], a[6], attnEps).reshaped([1, tokens, heads, d]),
                        qkv[2].reshaped([1, tokens, heads, d])]
            }
            // [canonical key, rope rows] -> [physical key]
            rotate = compile { a in [ropeReal(a[0], a[1])] }
            // [value, modulation, time_e0, query, cache_k, cache_v, o_w, o_b, cam, norm3_w, norm3_b, cq_w, cq_b,
            //  cross_norm_q, co_w, co_b, text_k, text_v, up_w, up_b, down_w, down_b] -> [value]
            post = compile { a in
                let tokens = a[0].dim(1)
                let e = modulation(a[1], a[2])
                var value = a[0]
                value = value + linear(attention(a[3], a[4], a[5]).reshaped([1, tokens, dim]), a[6], a[7]) * e[2]
                value = (1.0 + a[8][0]) * value + a[8][1]
                let q2 = rmsNorm(linear(layerNorm(value, norm3Eps, a[9], a[10]), a[11], a[12]), a[13], crossEps)
                value = value + linear(attention(q2.reshaped([1, tokens, heads, d]), a[16], a[17])
                                           .reshaped([1, tokens, dim]), a[14], a[15])
                let hidden = layerNorm(value, eps).asType(.float32) * (1 + e[4]) + e[3]
                let up = linear(hidden, a[18], a[19]).asType(.float32)
                let gelu = (0.5 * up * (1 + tanh(0.7978845608028654 * (up + 0.044715 * up * up * up))))
                    .asType(.bfloat16)
                return [value + linear(gelu, a[20], a[21]) * e[5]]
            }
        }

        func pass(_ message: Message) throws {
            let end = message.meta["end"] as! Int
            let timeE0 = message.arrays["time_e0"]!
            var value = message.arrays["value"]!
            let rows = end - tokens
            let ownRope = rope[start...], macRope = rope[..<start]
            // An idle GPU clocks down within milliseconds (RemoteStageServer); keep it busy while waiting
            let warmer = MLXRandom.normal([256, 256])
            let began = DispatchTime.now().uptimeNanoseconds
            var waited: UInt64 = 0
            for i in 0 ..< layerCount {
                let w = weights[i]
                let qkv = pre([value, w["modulation"]!, timeE0, w["qkv_weight"]!, w["qkv_bias"]!, w["norm_q"]!,
                               w["norm_k"]!, ownRope])
                eval(qkv)
                send(encode(op: "kv", layer: i, arrays: [("k", qkv[1]), ("v", qkv[2])]))
                cacheK[i][0..., (rows + start) ..< end] = rotate([qkv[1], ownRope])[0]
                cacheV[i][0..., (rows + start) ..< end] = qkv[2]
                let waitStart = DispatchTime.now().uptimeNanoseconds
                let theirs = try inbox.pop(idle: { eval(matmul(warmer, warmer)) })
                waited += DispatchTime.now().uptimeNanoseconds - waitStart
                guard theirs.op == "kv", theirs.layer == i else {
                    throw Failure(description: "expected kv \(i), got \(theirs.op) \(theirs.layer)")
                }
                cacheK[i][0..., rows ..< (rows + start)] = rotate([theirs.arrays["k"]!, macRope])[0]
                cacheV[i][0..., rows ..< (rows + start)] = theirs.arrays["v"]!
                let readK = cacheK[i][0..., ..<end], readV = cacheV[i][0..., ..<end]
                value = post([value, w["modulation"]!, timeE0, qkv[0], readK, readV, w["o_weight"]!, w["o_bias"]!,
                              camera[i], w["norm3_weight"]!, w["norm3_bias"]!, w["cross_q_weight"]!,
                              w["cross_q_bias"]!, w["cross_norm_q"]!, w["cross_o_weight"]!, w["cross_o_bias"]!,
                              textK[i], textV[i], w["ffn_up_weight"]!, w["ffn_up_bias"]!, w["ffn_down_weight"]!,
                              w["ffn_down_bias"]!])[0]
                eval(value, cacheK[i], cacheV[i])
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - began) / 1e6
            let waitMs = Double(waited) / 1e6
            let thermal = ["nominal", "fair", "serious", "critical"][ProcessInfo.processInfo.thermalState.rawValue]
            send(encode(op: "result", meta: ["ms": ms, "wait_ms": waitMs, "thermal": thermal],
                        arrays: [("value", value.asType(.float32))]))
            passes += 1
            if passes % 4 == 0 {
                log(String(format: "pass %d: %d tokens, %.0f ms here, %.0f ms of it waiting for the Mac's K/V; thermal %@",
                           passes, share, ms, waitMs, thermal))
            }
        }
    }
}
