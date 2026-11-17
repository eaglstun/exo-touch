import Foundation
import MLX
import os

/// exo's remote stage, served from this iPad: the Swift twin of
/// exo/src/exo/worker/engines/mlx/remote_stage/server.py, speaking protocol.py's frames.
/// A node connects, sends this stage's layers a layer at a time, then hidden states for them,
/// and keeps their KV caches here, named by handles it picks. Each connection has its own
/// thread, layers and caches: a proxy node asks for this device's memory on a connection of
/// its own while its runner's is busy.
enum RemoteStageServer {
    static let protocolVersion = 3

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
        var deviceModel = ""
        var memoryTotal = 0
        var memoryAvailable = 0
        var configJson = ""
        var layerCount = 0
        var kvBytesPerToken = 0
        var liveCaches = 0
        var message = ""
    }

    struct ConnectionClosed: Error {}

    /// Serve nodes on `port`, each connection on its own thread. A node sends the layers it
    /// wants run in `load` frames.
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
        guard bound == 0, listen(listener, 8) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        log("stage listening on port \(port)")

        while true {
            let connection = accept(listener, nil, nil)
            guard connection >= 0 else { continue }
            setsockopt(connection, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            Thread {
                let started = Date()
                let (forwards, handles) = serveConnection(connection, log: log)
                close(connection)
                // A memory poll says hello and goes: log only connections that did something
                if forwards > 0 || handles > 0 {
                    log(String(
                        format: "node disconnected after %.1fs: %d forwards, dropped %d caches",
                        Date().timeIntervalSince(started), forwards, handles))
                }
            }.start()
        }
    }

    /// This device's model identifier (e.g. iPad17,1)
    static func deviceModel() -> String {
        var system = utsname()
        uname(&system)
        return withUnsafeBytes(of: system.machine) { bytes in
            String(decoding: bytes.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    static func welcome() -> Frame {
        Frame(
            operation: "welcome", protocolVersion: protocolVersion, deviceModel: deviceModel(),
            memoryTotal: Int(ProcessInfo.processInfo.physicalMemory),
            // What this app can still allocate before iPadOS ends it: the memory a model may use
            memoryAvailable: Int(os_proc_available_memory()))
    }

    /// Serves one node until it disconnects. Returns the forwards run and the caches dropped.
    static func serveConnection(
        _ connection: Int32, log: @escaping (String) -> Void
    ) -> (Int, Int) {
        var layers: [LlamaShard.Layer] = []
        var ropeFrequencies = MLXArray(0)
        var kvBytesPerToken = 0
        // The layers a load has sent so far, and how many it will send
        var arriving: [String: MLXArray] = [:]
        var expectedLayers = 0
        var loadStarted = Date()
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
                    try send(welcome())
                case "load":
                    if frame.offset == 0 {
                        caches.removeAll()
                        layers = []
                        arriving = [:]
                        expectedLayers = frame.layerCount
                        loadStarted = Date()
                    }
                    let weights = try loadArrays(data: payload)
                    // Read now, so the frame's bytes can go
                    eval(Array(weights.values))
                    arriving.merge(weights) { _, new in new }
                    let received = Set(arriving.keys.compactMap { name -> Substring? in
                        let parts = name.split(separator: ".")
                        return parts.first == "layers" && parts.count > 1 ? parts[1] : nil
                    }).count
                    if received == expectedLayers {
                        let json = try JSONSerialization.jsonObject(with: Data(frame.configJson.utf8))
                        let config = LlamaShard.Config(json: json as? [String: Any] ?? [:])
                        (layers, ropeFrequencies) = try LlamaShard.loadLayers(arrays: arriving, config: config)
                        arriving = [:]
                        // Keys and values per token: layers x 2 x kv heads x head dim, in the
                        // activations' dtype
                        let itemSize = layers.first?.weights["input_layernorm.weight"]?.itemSize ?? 2
                        kvBytesPerToken = layers.count * 2 * config.kvHeads * config.headDim * itemSize
                        log(String(
                            format: "loaded %d layers in %.2fs; %.0f MB left to this app",
                            layers.count, Date().timeIntervalSince(loadStarted),
                            Double(os_proc_available_memory()) / 1e6))
                    }
                    try send(Frame(
                        operation: "loaded", layerCount: received,
                        kvBytesPerToken: layers.isEmpty ? 0 : kvBytesPerToken))
                case "new":
                    if !layers.isEmpty {
                        caches[frame.handle] = Array(repeating: LlamaShard.KVCache(), count: layers.count)
                    }
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
