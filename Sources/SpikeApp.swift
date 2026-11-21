import SwiftUI

@main
struct SpikeApp: App {
    init() {
        // Unbuffered stdout, so `devicectl device process launch --console` shows each line as it happens
        setvbuf(stdout, nil, _IONBF, 0)
    }

    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class SpikeLog: ObservableObject {
    @Published var lines: [String] = []
    @Published var running = false

    func add(_ line: String) {
        print("[spike] \(line)")
        lines.append(line)
    }
}

struct ContentView: View {
    @StateObject private var log = SpikeLog()

    var body: some View {
        NavigationStack {
            List(Array(log.lines.enumerated()), id: \.offset) { _, line in
                Text(line).font(.system(.body, design: .monospaced))
            }
            .navigationTitle("MLX ring spike")
            .toolbar {
                Button(log.running ? "Running…" : "Run") { start() }
                    .disabled(log.running)
            }
        }
        .onAppear {
            // Launched from the Mac with SPIKE_RANK and SPIKE_HOSTFILE_JSON: run at once
            let environment = ProcessInfo.processInfo.environment
            if environment["SPIKE_HOSTFILE_JSON"] != nil || ["stage", "dit", "ditblock", "split"].contains(environment["SPIKE_MODE"]) {
                start()
            }
        }
        // The ring needs the app in the foreground: keep the screen on while it runs
        .onChange(of: log.running) { _, running in
            UIApplication.shared.isIdleTimerDisabled = running
        }
    }

    private func start() {
        let environment = ProcessInfo.processInfo.environment
        if environment["SPIKE_MODE"] == "stage" {
            startStage(port: UInt16(environment["SPIKE_PORT"] ?? "") ?? 5560)
            return
        }
        if environment["SPIKE_MODE"] == "split" {
            // A share of FastFrames-LingBot's DiT tokens, in step with the Mac (lingbot_ipad_split.py)
            log.running = true
            let log = self.log
            let port = UInt16(environment["SPIKE_PORT"] ?? "") ?? SplitStage.defaultPort
            Task.detached {
                do {
                    try SplitStage.serve(port: port) { line in Task { @MainActor in log.add(line) } }
                } catch {
                    Task { @MainActor in log.add("FAILED: \(error)") }
                }
                Task { @MainActor in log.running = false }
            }
            return
        }
        if environment["SPIKE_MODE"] == "ditblock" {
            // FastFrames-LingBot's real DiT block, checked against PyTorch's output (no ring)
            log.running = true
            let log = self.log
            let name = environment["SPIKE_BLOCK"] ?? "block0"
            Task.detached {
                do {
                    try DiTBlockCheck.run(name: name) { line in Task { @MainActor in log.add(line) } }
                } catch {
                    Task { @MainActor in log.add("FAILED: \(error)") }
                }
                Task { @MainActor in log.running = false }
            }
            return
        }
        if environment["SPIKE_MODE"] == "dit" {
            // FastFrames-LingBot DiT block shapes on this GPU (no ring, no Mac rank)
            log.running = true
            let log = self.log
            Task.detached {
                DiTBench.run { line in Task { @MainActor in log.add(line) } }
                Task { @MainActor in log.running = false }
            }
            return
        }
        guard let hostfile = environment["SPIKE_HOSTFILE_JSON"] else {
            log.add("No SPIKE_HOSTFILE_JSON: launch from the Mac with mac/run_spike.sh")
            return
        }
        let rank = Int(environment["SPIKE_RANK"] ?? "1") ?? 1
        log.running = true
        let log = self.log
        let llama = environment["SPIKE_MODE"] == "llama"
        let shard = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ipad_shard.safetensors")
        Task.detached {
            do {
                let add: (String) -> Void = { line in Task { @MainActor in log.add(line) } }
                if llama {
                    try LlamaShard.run(rank: rank, hostfileJSON: hostfile, shard: shard, log: add)
                } else {
                    try RingSpike.run(rank: rank, hostfileJSON: hostfile, log: add)
                }
            } catch {
                Task { @MainActor in log.add("FAILED: \(error)") }
            }
            Task { @MainActor in log.running = false }
        }
    }

    /// Serve exo's remote stage protocol on `port` until the app quits
    private func startStage(port: UInt16) {
        log.running = true
        let log = self.log
        Task.detached {
            do {
                try RemoteStageServer.serve(port: port) { line in
                    Task { @MainActor in log.add(line) }
                }
            } catch {
                Task { @MainActor in log.add("FAILED: \(error)") }
            }
            Task { @MainActor in log.running = false }
        }
    }
}
