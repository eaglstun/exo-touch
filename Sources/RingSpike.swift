import Foundation
import MLX

/// The iPad's half of the spike: rank 1 of a two-rank MLX ring whose rank 0 is
/// mac/mac_rank.py. Every step mirrors a step there, in the same order.
enum RingSpike {
    /// One hidden state of Llama-3.2-1B for a 12-token prompt: what a pipeline sends in prefill
    static let prefillShape = [1, 12, 2048]
    /// One token's hidden state: what a pipeline sends per decoded token
    static let decodeShape = [1, 1, 2048]
    static let roundTrips = 100
    /// 16 MB of float32 per message
    static let bandwidthElements = 4 * 1024 * 1024
    static let bandwidthMessages = 20

    static func run(rank: Int, hostfileJSON: String, log: @escaping (String) -> Void) throws {
        let hostfile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mlx-hostfile.json")
        try hostfileJSON.write(to: hostfile, atomically: true, encoding: .utf8)
        setenv("MLX_HOSTFILE", hostfile.path, 1)
        setenv("MLX_RANK", String(rank), 1)

        log("device: \(Device.defaultDevice()), ring available: \(MLXDistributed.isAvailable(.ring))")
        log("hostfile: \(hostfileJSON)")

        // 1. Join the ring
        var started = Date()
        let group = try MLXDistributed.initialize(backend: .ring, strict: true)
        let ownRank = try group.rank
        let size = try group.size
        log("joined: rank \(ownRank) of \(size) in \(seconds(since: started))")
        let peer = 1 - ownRank

        // 2. A collective: each rank adds rank + 1, so two ranks give 3
        let summed = MLXDistributed.allSum(ones([4]) * Float(ownRank + 1), group: group)
        eval(summed)
        log("allSum: \(summed.asArray(Float.self)) (expect 3s)")

        // 3. A prefill-shaped float16 hidden state: receive it, double it on this device's
        //    GPU, send it back. Rank 0 checks the values.
        started = Date()
        let hidden = MLXDistributed.recv(prefillShape, dtype: .float16, from: peer, group: group)
        eval(hidden)
        let doubled = (hidden * 2).asType(.float16)
        eval(doubled)
        eval(MLXDistributed.send(doubled, to: peer, group: group))
        log("prefill hop: received \(prefillShape) float16, doubled on GPU, sent back in \(seconds(since: started))")

        // 4. Decode-shaped round trips: the per-token cost a pipeline stage adds
        started = Date()
        for _ in 0 ..< roundTrips {
            let token = MLXDistributed.recv(decodeShape, dtype: .float16, from: peer, group: group)
            eval(token)
            eval(MLXDistributed.send(token, to: peer, group: group))
        }
        let roundTripMilliseconds = Date().timeIntervalSince(started) * 1000 / Double(roundTrips)
        log(String(format: "decode round trip: %.3f ms average over %d", roundTripMilliseconds, roundTrips))

        // 5. Bandwidth: receive 20 messages of 16 MB
        started = Date()
        for _ in 0 ..< bandwidthMessages {
            eval(MLXDistributed.recv([bandwidthElements], dtype: .float32, from: peer, group: group))
        }
        let elapsed = Date().timeIntervalSince(started)
        let megabytes = Double(bandwidthMessages * bandwidthElements * 4) / 1_000_000
        log(String(format: "bandwidth: %.0f MB in %.2f s = %.0f MB/s", megabytes, elapsed, megabytes / elapsed))

        // A last collective, so neither rank exits while the other is still receiving
        eval(MLXDistributed.allSum(ones([1]), group: group))
        log("DONE")
    }

    private static func seconds(since start: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(start))
    }
}
