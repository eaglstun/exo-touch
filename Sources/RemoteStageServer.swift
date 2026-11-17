import Foundation
import MLX

/// exo's remote stage, served from this iPad: the Swift twin of
/// exo/src/exo/worker/engines/mlx/remote_stage/server.py, speaking protocol.py's frames.
/// A node connects, sends hidden states for this stage's layers and keeps their KV caches
/// here, named by handles it picks.
enum RemoteStageServer {
    static let protocolVersion = 1

    /// protocol.py's Frame: every field, snake_case on the wire
    struct Frame: Codable {
        var operation: String
        var handle = 0
        var source = 0
        var offset = 0
        var shape: [Int] = []
        var dtype = ""
        var payloadBytes = 0
        var protocolVersion = 0
        var layerCount = 0
        var kvBytesPerToken = 0
        var liveCaches = 0
        var message = ""
    }

    struct ConnectionClosed: Error {}

    static func serve(port: UInt16, shard: URL, log: @escaping (String) -> Void) throws {
        var started = Date()
        let (layers, ropeFrequencies) = try LlamaShard.loadLayers(shard: shard)
        log(String(format: "loaded %d layers in %.2fs", layers.count, Date().timeIntervalSince(started)))

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
        guard bound == 0, listen(listener, 1) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        log("stage listening on port \(port)")

        while true {
            let connection = accept(listener, nil, nil)
            guard connection >= 0 else { continue }
            setsockopt(connection, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            log("node connected")
            started = Date()
            let (forwards, handles) = serveConnection(
                connection, layers: layers, ropeFrequencies: ropeFrequencies)
            close(connection)
            log(String(
                format: "node disconnected after %.1fs: %d forwards, dropped %d caches",
                Date().timeIntervalSince(started), forwards, handles))
        }
    }

    /// Serves one node until it disconnects. Returns the forwards run and the caches dropped.
    static func serveConnection(
        _ connection: Int32, layers: [LlamaShard.Layer], ropeFrequencies: MLXArray
    ) -> (Int, Int) {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var caches: [Int: [LlamaShard.KVCache]] = [:]
        var forwards = 0

        func send(_ frame: Frame, _ payload: Data = Data()) throws {
            let header = try encoder.encode(frame)
            var length = UInt32(header.count).littleEndian
            var message = Data(bytes: &length, count: 4)
            message.append(header)
            message.append(payload)
            try writeAll(connection, message)
        }

        do {
            while true {
                let length = try readExactly(connection, 4).withUnsafeBytes {
                    UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
                }
                let frame = try decoder.decode(Frame.self, from: try readExactly(connection, Int(length)))
                let payload = try readExactly(connection, frame.payloadBytes)

                switch frame.operation {
                case "hello":
                    // Keys and values per token: layers x 2 x kv heads x head dim, float16
                    try send(Frame(
                        operation: "welcome", protocolVersion: protocolVersion, layerCount: layers.count,
                        kvBytesPerToken: layers.count * 2 * LlamaShard.kvHeads * LlamaShard.headDim * 2))
                case "new":
                    caches[frame.handle] = Array(repeating: LlamaShard.KVCache(), count: layers.count)
                case "copy":
                    // MLX arrays never change in place, so a copy shares them
                    caches[frame.handle] = caches[frame.source]
                case "free":
                    caches[frame.handle] = nil
                case "truncate":
                    caches[frame.handle] = caches[frame.handle]?.map { $0.truncated(to: frame.offset) }
                case "stats":
                    try send(Frame(operation: "counted", liveCaches: caches.count))
                case "forward":
                    guard var layerCaches = caches[frame.handle], let dtype = dtype(named: frame.dtype) else {
                        try send(Frame(
                            operation: "error",
                            message: "no cache \(frame.handle) or dtype \(frame.dtype) on the stage"))
                        continue
                    }
                    var hidden = MLXArray(payload, frame.shape, dtype: dtype)
                    for index in layers.indices {
                        hidden = LlamaShard.forward(
                            hidden, layer: layers[index], cache: &layerCaches[index],
                            ropeFrequencies: ropeFrequencies)
                    }
                    hidden = hidden.asType(dtype)
                    eval(hidden)
                    eval(layerCaches.flatMap { [$0.keys, $0.values].compactMap { $0 } })
                    caches[frame.handle] = layerCaches
                    let data = hidden.asData(access: .copy).data
                    try send(
                        Frame(
                            operation: "hidden", handle: frame.handle, shape: frame.shape, dtype: frame.dtype,
                            payloadBytes: data.count),
                        data)
                    forwards += 1
                default:
                    try send(Frame(operation: "error", message: "a node doesn't send \(frame.operation)"))
                    return (forwards, caches.count)
                }
            }
        } catch {
            return (forwards, caches.count)
        }
    }

    static func dtype(named name: String) -> DType? {
        switch name {
        case "float16": .float16
        case "bfloat16": .bfloat16
        case "float32": .float32
        default: nil
        }
    }

    static func readExactly(_ connection: Int32, _ count: Int) throws -> Data {
        var data = Data(count: count)
        var received = 0
        while received < count {
            let result = data.withUnsafeMutableBytes {
                recv(connection, $0.baseAddress! + received, count - received, 0)
            }
            if result <= 0 { throw ConnectionClosed() }
            received += result
        }
        return data
    }

    static func writeAll(_ connection: Int32, _ data: Data) throws {
        var sent = 0
        while sent < data.count {
            let result = data.withUnsafeBytes {
                Darwin.send(connection, $0.baseAddress! + sent, data.count - sent, 0)
            }
            if result <= 0 { throw ConnectionClosed() }
            sent += result
        }
    }
}
