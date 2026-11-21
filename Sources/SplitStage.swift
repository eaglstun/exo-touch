import Foundation
import MLX

/// This iPad's share of one FastFrames-LingBot DiT denoising pass, in step with a Mac running the
/// rest in PyTorch (fastframes' scripts/lingbot_ipad_split.py). The chunk's tokens are split: the Mac
/// keeps [0, start), this device runs [start, tokens) through all 30 blocks. In each block both
/// sides compute their tokens' keys and values, swap them, write both into their own copy of the
/// KV cache and attend over all of it; the rest of the block is per token. The block math is
/// DiTBlockCheck's (checked against PyTorch).
///
/// Frames both ways: little-endian u32 header length, a JSON header, then the arrays' bytes in the
/// header's order: {"op", "layer", "meta", "arrays": [{"name", "shape", "dtype"}]}.
///   Mac -> iPad: setup (meta, rope, value), layer (weights, caches, text K/V, camera rows), pass
///                (time_e0, value), kv (the Mac's keys and values of a layer), bye
///   iPad -> Mac: ready, kv (this device's), result (this device's tokens after the last block)
enum SplitStage {
    static let defaultPort: UInt16 = 5570

    struct Message {
        var op: String
        var layer = 0
        var meta: [String: Any] = [:]
        var arrays: [String: MLXArray] = [:]
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
                try serveConnection(connection, log: log)
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

    static func itemSize(_ dtype: DType) -> Int {
        switch dtype {
        case .float32: 4
        default: 2
        }
    }

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
            let bytes = shape.reduce(1, *) * itemSize(dtype)
            message.arrays[spec["name"] as! String] = MLXArray(
                try RemoteStageServer.readExactly(connection, bytes), shape, dtype: dtype)
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

    // MARK: the pass

    struct Layer {
        var w: [String: MLXArray]
        var cacheK: MLXArray
        var cacheV: MLXArray
    }

    static func serveConnection(_ connection: Int32, log: @escaping @Sendable (String) -> Void) throws {
        let inbox = Inbox()
        Thread {
            while true {
                do { inbox.push(try read(connection)) } catch { inbox.push(nil); return }
            }
        }.start()
        let sender = DispatchQueue(label: "split-sender")
        func send(_ data: Data) { sender.async { try? RemoteStageServer.writeAll(connection, data) } }

        var setup = try inbox.pop()
        guard setup.op == "setup" else { throw NSError(domain: "SplitStage", code: 1) }
        let meta = setup.meta
        let tokens = meta["tokens"] as! Int, start = meta["start"] as! Int, end = meta["end"] as! Int
        let heads = meta["heads"] as! Int, layerCount = meta["layers"] as! Int
        let eps = Float(meta["eps"] as! Double), attnEps = Float(meta["attn_eps"] as! Double)
        let crossEps = Float(meta["cross_eps"] as! Double), norm3Eps = Float(meta["norm3_eps"] as! Double)
        let rope = setup.arrays["rope"]!
        let share = tokens - start
        let dim = setup.arrays["value"]!.dim(-1), d = dim / heads
        let scale = 1 / Float(d).squareRoot()
        setup.arrays = [:]
        log("setup: tokens \(start)..<\(tokens) of \(tokens), cache rows \(end), \(layerCount) layers")

        var layers: [Layer] = []
        let loading = Date()
        for _ in 0 ..< layerCount {
            var message = try inbox.pop()
            guard message.op == "layer" else { throw NSError(domain: "SplitStage", code: 2) }
            let cacheK = message.arrays.removeValue(forKey: "cache_k")!
            let cacheV = message.arrays.removeValue(forKey: "cache_v")!
            eval(Array(message.arrays.values) + [cacheK, cacheV])
            layers.append(Layer(w: message.arrays, cacheK: cacheK, cacheV: cacheV))
        }
        log(String(format: "loaded %d layers in %.1f s; app footprint %.1f GB", layerCount,
                   Date().timeIntervalSince(loading), Double(RemoteStageServer.footprint()) / 1e9))

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
        func ropeReal(_ x: MLXArray) -> MLXArray {
            let shape = x.shape
            let pairs = x.asType(.float32).reshaped(shape.dropLast() + [d / 2, 2])
            let xr = pairs[.ellipsis, 0], xi = pairs[.ellipsis, 1]
            let fr = rope[.ellipsis, 0], fi = rope[.ellipsis, 1]
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
        // [value, modulation, time_e0, qkv_weight, qkv_bias, norm_q, norm_k] -> [query, key, value rows]
        let pre = compile { (a: [MLXArray]) -> [MLXArray] in
            let e = modulation(a[1], a[2])
            let hidden = layerNorm(a[0], eps).asType(.float32) * (1 + e[1]) + e[0]
            let qkv = split(linear(hidden, a[3], a[4]), parts: 3, axis: -1)
            return [ropeReal(rmsNorm(qkv[0], a[5], attnEps).reshaped([1, share, heads, d])),
                    ropeReal(rmsNorm(qkv[1], a[6], attnEps).reshaped([1, share, heads, d])),
                    qkv[2].reshaped([1, share, heads, d])]
        }
        // [value, modulation, time_e0, query, cache_k, cache_v, o_w, o_b, cam, norm3_w, norm3_b, cq_w, cq_b,
        //  cross_norm_q, co_w, co_b, text_k, text_v, up_w, up_b, down_w, down_b] -> [value]
        let post = compile { (a: [MLXArray]) -> [MLXArray] in
            let e = modulation(a[1], a[2])
            var value = a[0]
            value = value + linear(attention(a[3], a[4], a[5]).reshaped([1, share, dim]), a[6], a[7]) * e[2]
            value = (1.0 + a[8][0]) * value + a[8][1]
            let q2 = rmsNorm(linear(layerNorm(value, norm3Eps, a[9], a[10]), a[11], a[12]), a[13], crossEps)
            value = value + linear(attention(q2.reshaped([1, share, heads, d]), a[16], a[17])
                                       .reshaped([1, share, dim]), a[14], a[15])
            let hidden = layerNorm(value, eps).asType(.float32) * (1 + e[4]) + e[3]
            let up = linear(hidden, a[18], a[19]).asType(.float32)
            let gelu = (0.5 * up * (1 + tanh(0.7978845608028654 * (up + 0.044715 * up * up * up)))).asType(.bfloat16)
            return [value + linear(gelu, a[20], a[21]) * e[5]]
        }
        send(encode(op: "ready", meta: ["footprint": RemoteStageServer.footprint()], arrays: []))

        // An idle GPU clocks down within milliseconds (RemoteStageServer); keep it busy while waiting
        let warmer = MLXRandom.normal([256, 256])
        let ownRows = (end - tokens + start) ..< end, macRows = (end - tokens) ..< (end - tokens + start)
        while true {
            let message = try inbox.pop()
            if message.op == "bye" { return }
            guard message.op == "pass" else { throw NSError(domain: "SplitStage", code: 3) }
            let timeE0 = message.arrays["time_e0"]!
            var value = message.arrays["value"]!
            let started = DispatchTime.now().uptimeNanoseconds
            var waited: UInt64 = 0
            for i in 0 ..< layerCount {
                let w = layers[i].w
                let qkv = pre([value, w["modulation"]!, timeE0, w["qkv_weight"]!, w["qkv_bias"]!, w["norm_q"]!,
                               w["norm_k"]!])
                eval(qkv)
                send(encode(op: "kv", layer: i, arrays: [("k", qkv[1]), ("v", qkv[2])]))
                layers[i].cacheK[0..., ownRows] = qkv[1]
                layers[i].cacheV[0..., ownRows] = qkv[2]
                let waitStart = DispatchTime.now().uptimeNanoseconds
                let theirs = try inbox.pop(idle: { eval(matmul(warmer, warmer)) })
                waited += DispatchTime.now().uptimeNanoseconds - waitStart
                guard theirs.op == "kv", theirs.layer == i else { throw NSError(domain: "SplitStage", code: 4) }
                layers[i].cacheK[0..., macRows] = theirs.arrays["k"]!
                layers[i].cacheV[0..., macRows] = theirs.arrays["v"]!
                value = post([value, w["modulation"]!, timeE0, qkv[0], layers[i].cacheK, layers[i].cacheV,
                              w["o_weight"]!, w["o_bias"]!, w["cam"]!, w["norm3_weight"]!, w["norm3_bias"]!,
                              w["cross_q_weight"]!, w["cross_q_bias"]!, w["cross_norm_q"]!, w["cross_o_weight"]!,
                              w["cross_o_bias"]!, w["text_k"]!, w["text_v"]!, w["ffn_up_weight"]!,
                              w["ffn_up_bias"]!, w["ffn_down_weight"]!, w["ffn_down_bias"]!])[0]
                eval(value, layers[i].cacheK, layers[i].cacheV)
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            let waitMs = Double(waited) / 1e6
            send(encode(op: "result", meta: ["ms": ms, "wait_ms": waitMs], arrays: [("value", value)]))
            log(String(format: "pass: %.0f ms on this device, %.0f ms of it waiting for the Mac's K/V", ms, waitMs))
        }
    }
}
