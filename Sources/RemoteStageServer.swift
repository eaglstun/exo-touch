import Foundation
import Metal
import MLX
import os

/// exo's remote stage, served from this iPad: the Swift twin of
/// exo/src/exo/worker/engines/mlx/remote_stage/server.py, speaking protocol.py's frames.
/// A node connects, sends this stage's layers a layer at a time, then hidden states for them,
/// and keeps their KV caches here, named by handles it picks. Each connection has its own
/// thread, layers and caches: a proxy node asks for this device's memory on a connection of
/// its own while its runner's is busy.
enum RemoteStageServer {
    static let protocolVersion = 5

    /// protocol.py's Frame: every field, snake_case on the wire
    struct Frame: Codable {
        var operation: String
        var handle = 0
        var source = 0
        var sources: [Int] = []
        var rows: [Int] = []
        var leftPadding: [Int] = []
        var rightPadding: [Int] = []
        var offset = 0
        var shape: [Int] = []
        var dtype = ""
        var payloadBytes = 0
        var protocolVersion = 0
        var deviceModel = ""
        var memoryTotal = 0
        var memoryAvailable = 0
        var gpuWorkingSet = 0
        var configJson = ""
        var layerCount = 0
        var kvBytesPerToken = 0
        var stateBytes = 0
        var liveCaches = 0
        var message = ""
    }

    struct ConnectionClosed: Error {}

    /// One layer's cache of one sequence: keys and values, or a linear attention layer's
    /// recurrent state
    enum SequenceCache {
        case attention(LlamaShard.KVCache)
        case linear(LlamaShard.RecurrentCache)
    }

    /// One layer's cache of a batch (mlx-lm keeps a linear attention layer's state in the
    /// same ArraysCache for a batch as for one sequence)
    enum BatchCache {
        case attention(LlamaShard.BatchKVCache)
        case linear(LlamaShard.RecurrentCache)
    }

    /// What a handle names: one sequence's caches, or a batch's, one per layer
    enum Caches {
        case single([SequenceCache])
        case batch([BatchCache])
    }

    /// Run `layers` over `hidden`, each with its cache
    static func forward(
        _ hidden: MLXArray, layers: [LlamaShard.Layer], caches: inout [SequenceCache],
        ropeFrequencies: MLXArray
    ) -> MLXArray {
        var hidden = hidden
        for index in layers.indices {
            switch caches[index] {
            case .attention(var cache):
                hidden = LlamaShard.forward(hidden, layer: layers[index], cache: &cache, ropeFrequencies: ropeFrequencies)
                caches[index] = .attention(cache)
            case .linear(var cache):
                hidden = LlamaShard.forward(hidden, layer: layers[index], cache: &cache)
                caches[index] = .linear(cache)
            }
        }
        return hidden
    }

    static func forward(
        _ hidden: MLXArray, layers: [LlamaShard.Layer], caches: inout [BatchCache],
        ropeFrequencies: MLXArray
    ) -> MLXArray {
        var hidden = hidden
        for index in layers.indices {
            switch caches[index] {
            case .attention(var cache):
                hidden = LlamaShard.forward(hidden, layer: layers[index], cache: &cache, ropeFrequencies: ropeFrequencies)
                caches[index] = .attention(cache)
            case .linear(var cache):
                hidden = LlamaShard.forward(hidden, layer: layers[index], cache: &cache)
                caches[index] = .linear(cache)
            }
        }
        return hidden
    }

    /// Every array the caches hold: evaluated after each forward, so a copy of a cache holds
    /// arrays, not a graph that grows (nothing else in a step reads the convolution's state)
    static func arrays(_ caches: Caches) -> [MLXArray] {
        switch caches {
        case .single(let layers):
            layers.flatMap { cache -> [MLXArray] in
                switch cache {
                case .attention(let kv): [kv.keys, kv.values].compactMap { $0 }
                case .linear(let recurrent): recurrent.arrays
                }
            }
        case .batch(let layers):
            layers.flatMap { cache -> [MLXArray] in
                switch cache {
                case .attention(let kv): [kv.keys, kv.values].compactMap { $0 }
                case .linear(let recurrent): recurrent.arrays
                }
            }
        }
    }

    /// The same change to each layer's batch cache, by kind
    static func eachBatch(
        _ caches: [BatchCache], attention: (inout LlamaShard.BatchKVCache) -> Void,
        linear: (inout LlamaShard.RecurrentCache) -> Void
    ) -> [BatchCache] {
        caches.map {
            switch $0 {
            case .attention(var cache):
                attention(&cache)
                return .attention(cache)
            case .linear(var cache):
                linear(&cache)
                return .linear(cache)
            }
        }
    }

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

    /// This app's memory footprint: what iPadOS counts against its limit
    static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    static func welcome() -> Frame {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        return Frame(
            operation: "welcome", protocolVersion: protocolVersion, deviceModel: deviceModel(),
            memoryTotal: physical,
            // What this app can still allocate before iPadOS ends it. With the increased memory
            // limit that allowance can exceed the RAM (it counts compressed memory), so never
            // more than the RAM this app doesn't use yet
            memoryAvailable: min(Int(os_proc_available_memory()), physical - footprint()),
            // The most Metal says the GPU should use, as exo reads it on a Mac
            gpuWorkingSet: Int(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0))
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
        var caches: [Int: Caches] = [:]

        var stateBytes = 0
        func batch(_ handle: Int) -> [BatchCache] {
            guard case .batch(let layers) = caches[handle] else { return [] }
            return layers
        }
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
                    let json = try JSONSerialization.jsonObject(with: Data(frame.configJson.utf8))
                    let config = LlamaShard.Config(json: json as? [String: Any] ?? [:])
                    guard LlamaShard.modelTypes.contains(config.modelType) else {
                        try send(Frame(
                            operation: "error",
                            message: "this stage can't run \(config.modelType) layers"))
                        continue
                    }
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
                        (layers, ropeFrequencies) = try LlamaShard.loadLayers(arrays: arriving, config: config)
                        arriving = [:]
                        // Keys and values per token: layers x 2 x kv heads x head dim, in the
                        // activations' dtype
                        let itemSize = layers.first?.weights["input_layernorm.weight"]?.itemSize ?? 2
                        let attentionLayers = layers.filter { !$0.isLinear }.count
                        kvBytesPerToken = attentionLayers * 2 * config.kvHeads * config.headDim * itemSize
                        // A linear attention layer's state: (kernel - 1) convolution inputs in
                        // the activations' dtype, and a float32 state per value head
                        let convWidth = 2 * config.linearKeyHeads * config.linearKeyDim
                            + config.linearValueHeads * config.linearValueDim
                        let perLinearLayer = (config.convKernel - 1) * convWidth * itemSize
                            + config.linearValueHeads * config.linearValueDim * config.linearKeyDim * 4
                        stateBytes = (layers.count - attentionLayers) * perLinearLayer
                        log(String(
                            format: "loaded %d layers in %.2fs; %.0f MB left to this app",
                            layers.count, Date().timeIntervalSince(loadStarted),
                            Double(os_proc_available_memory()) / 1e6))
                    }
                    try send(Frame(
                        operation: "loaded", layerCount: received,
                        kvBytesPerToken: layers.isEmpty ? 0 : kvBytesPerToken,
                        stateBytes: layers.isEmpty ? 0 : stateBytes))
                case "new":
                    if !layers.isEmpty {
                        caches[frame.handle] = .single(layers.map {
                            $0.isLinear ? .linear(LlamaShard.RecurrentCache()) : .attention(LlamaShard.KVCache())
                        })
                    }
                case "copy":
                    // MLX arrays never change in place, so a copy shares them
                    caches[frame.handle] = caches[frame.source]
                case "free":
                    caches[frame.handle] = nil
                case "truncate":
                    // Only a stage without recurrent state is sent truncate: a node goes back
                    // to a copy instead
                    if case .single(let layerCaches) = caches[frame.handle] {
                        caches[frame.handle] = .single(layerCaches.map {
                            guard case .attention(let cache) = $0 else { return $0 }
                            return .attention(cache.truncated(to: frame.offset))
                        })
                    }
                case "merge":
                    let sequences = frame.sources.compactMap { source -> [SequenceCache]? in
                        guard case .single(let layerCaches) = caches[source] else { return nil }
                        return layerCaches
                    }
                    caches[frame.handle] = .batch(layers.indices.map { layer -> BatchCache in
                        if layers[layer].isLinear {
                            return .linear(LlamaShard.RecurrentCache.merge(sequences.map {
                                guard case .linear(let cache) = $0[layer] else { return LlamaShard.RecurrentCache() }
                                return cache
                            }))
                        }
                        return .attention(LlamaShard.BatchKVCache.merge(sequences.map {
                            guard case .attention(let cache) = $0[layer] else { return LlamaShard.KVCache() }
                            return cache
                        }))
                    })
                case "extend":
                    let other = batch(frame.source)
                    caches[frame.handle] = .batch(zip(batch(frame.handle), other).map { own, theirs in
                        switch (own, theirs) {
                        case (.attention(var cache), .attention(let other)):
                            cache.extend(other)
                            return .attention(cache)
                        case (.linear(var cache), .linear(let other)):
                            cache.extend(other)
                            return .linear(cache)
                        default:
                            return own
                        }
                    })
                case "filter":
                    caches[frame.handle] = .batch(eachBatch(
                        batch(frame.handle), attention: { $0.filter(frame.rows) }, linear: { $0.filter(frame.rows) }))
                case "extract":
                    caches[frame.handle] = .single(batch(frame.source).map {
                        switch $0 {
                        case .attention(let cache): .attention(cache.extract(frame.rows[0]))
                        case .linear(let cache): .linear(cache.extract(frame.rows[0]))
                        }
                    })
                case "prepare":
                    // A linear attention layer's state needs nothing for left padding, the only
                    // padding exo makes (its prepare takes lengths, for right padding)
                    caches[frame.handle] = .batch(eachBatch(
                        batch(frame.handle),
                        attention: { $0.prepare(leftPadding: frame.leftPadding, rightPadding: frame.rightPadding) },
                        linear: { _ in }))
                case "finalize":
                    caches[frame.handle] = .batch(eachBatch(
                        batch(frame.handle), attention: { $0.finalize() }, linear: { $0.finalize() }))
                case "trim":
                    switch caches[frame.handle] {
                    case .single(let layerCaches):
                        caches[frame.handle] = .single(layerCaches.map {
                            guard case .attention(let cache) = $0 else { return $0 }
                            return .attention(cache.truncated(to: max(0, cache.offset - frame.offset)))
                        })
                    case .batch(let layerCaches):
                        caches[frame.handle] = .batch(eachBatch(
                            layerCaches, attention: { $0.trim(frame.offset) }, linear: { _ in }))
                    case nil:
                        break
                    }
                case "stats":
                    let device = welcome()
                    try send(Frame(
                        operation: "counted", memoryTotal: device.memoryTotal,
                        memoryAvailable: device.memoryAvailable, liveCaches: caches.count))
                case "forward":
                    guard let entry = caches[frame.handle], let dtype = dtype(named: frame.dtype) else {
                        try send(Frame(
                            operation: "error",
                            message: "no cache \(frame.handle) or dtype \(frame.dtype) on the stage"))
                        continue
                    }
                    var hidden = MLXArray(payload, frame.shape, dtype: dtype)
                    switch entry {
                    case .single(var layerCaches):
                        hidden = forward(hidden, layers: layers, caches: &layerCaches, ropeFrequencies: ropeFrequencies)
                        caches[frame.handle] = .single(layerCaches)
                    case .batch(var layerCaches):
                        hidden = forward(hidden, layers: layers, caches: &layerCaches, ropeFrequencies: ropeFrequencies)
                        caches[frame.handle] = .batch(layerCaches)
                    }
                    hidden = hidden.asType(dtype)
                    eval([hidden] + arrays(caches[frame.handle]!))
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
