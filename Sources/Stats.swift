import Darwin
import Foundation
import MLX
import Synchronization

/// What the stats panel shows. The servers write it from their own threads; the panel copies it
/// a few times a second and works out rates from the difference.
enum Stats {
    struct State {
        var mode = ""
        var port: UInt16 = 0
        /// The node (or Mac) being served now, and since when
        var peer: String?
        var connectedAt: Date?

        // The model
        var modelName = ""
        var modelDetail = ""
        var layersLoaded = 0
        var layersExpected = 0
        var weightBytes = 0

        // exo's remote stage: forwards, and the tokens they carried
        var forwards = 0
        var decodeTokens = 0
        var prefillTokens = 0
        var lastForwardMs = 0.0
        var lastForwardAt: Date?
        var liveCaches = 0
        var cacheBytes = 0
        var kvBytesPerToken = 0
        var keepWarm = false

        // LingBot's token split: DiT passes
        var passes = 0
        var passMs = 0.0
        var waitMs = 0.0
        var shareTokens = 0
        var chunkTokens = 0
        var cacheRows = 0

        // Every byte through the sockets, either mode
        var bytesIn = 0
        var bytesOut = 0
    }

    private static let state = Mutex(State())

    static func update(_ change: (inout State) -> Void) {
        state.withLock { change(&$0) }
    }

    static func snapshot() -> State {
        state.withLock { $0 }
    }

    /// The address on the far side of a connected socket, e.g. 169.254.179.218
    static func peerAddress(_ connection: Int32) -> String? {
        var address = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getpeername(connection, $0, &length) }
        }
        guard result == 0 else { return nil }
        var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        var raw = address.sin_addr
        guard inet_ntop(AF_INET, &raw, &text, socklen_t(INET_ADDRSTRLEN)) != nil else { return nil }
        return String(cString: text)
    }

    /// A name for the model exo sent: the config's own if it has one, else its architecture
    /// (exo sends config.json, which names the class, not the checkpoint)
    static func modelName(config: [String: Any]) -> String {
        for key in ["exo_model_id", "model_id", "_name_or_path", "name_or_path"] {
            if let name = config[key] as? String, !name.isEmpty {
                return name.split(separator: "/").last.map(String.init) ?? name
            }
        }
        if let architecture = (config["architectures"] as? [String])?.first {
            for suffix in ["ForCausalLM", "ForConditionalGeneration", "Model"] where architecture.hasSuffix(suffix) {
                return String(architecture.dropLast(suffix.count))
            }
            return architecture
        }
        return (config["model_type"] as? String ?? "model").capitalized
    }

    /// e.g. "llama · 4-bit · hidden 2048"
    static func modelDetail(config: [String: Any]) -> String {
        let type = config["model_type"] as? String ?? "?"
        let precision: String
        if let bits = (config["quantization"] as? [String: Any])?["bits"] as? Int {
            precision = "\(bits)-bit"
        } else {
            precision = (config["torch_dtype"] as? String ?? config["dtype"] as? String ?? "bf16")
                .replacingOccurrences(of: "bfloat16", with: "bf16")
                .replacingOccurrences(of: "float16", with: "fp16")
        }
        var parts = [type, precision]
        if let hidden = config["hidden_size"] as? Int { parts.append("hidden \(hidden)") }
        if let experts = config["num_experts"] as? Int ?? config["n_routed_experts"] as? Int, experts > 0 {
            parts.append("\(experts) experts")
        }
        return parts.joined(separator: " · ")
    }

    /// Bytes held by every cache handle of the stage
    static func cacheBytes(_ caches: [Int: RemoteStageServer.Caches]) -> Int {
        caches.values.reduce(0) { total, entry in
            total + RemoteStageServer.arrays(entry).reduce(0) { $0 + $1.nbytes }
        }
    }
}
