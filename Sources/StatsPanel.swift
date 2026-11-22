import Charts
import MLX
import SwiftUI

/// Copies `Stats` twice a second, and turns its counters into rates and a minute of history
@MainActor
final class StatsModel: ObservableObject {
    @Published private(set) var state = Stats.State()
    @Published private(set) var tokensPerSecond = 0.0
    @Published private(set) var prefillPerSecond = 0.0
    @Published private(set) var inPerSecond = 0.0
    @Published private(set) var outPerSecond = 0.0
    @Published private(set) var tokenHistory: [Double] = []
    @Published private(set) var linkHistory: [Double] = []
    @Published private(set) var footprint = 0
    @Published private(set) var available = 0
    @Published private(set) var gpuActive = 0
    @Published private(set) var thermal = ProcessInfo.ThermalState.nominal

    static let interval = 0.5
    static let historyLength = 120  // a minute

    private var previous: (state: Stats.State, at: Date)?
    private var timer: Timer?

    func start() {
        guard timer == nil else { return }
        tick()
        timer = Timer.scheduledTimer(withTimeInterval: Self.interval, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func tick() {
        let now = Date()
        let current = Stats.snapshot()
        if let previous {
            let seconds = max(now.timeIntervalSince(previous.at), 0.05)
            // Smoothed: one window of a decode run can catch a node's own layers, or a pause
            func smooth(_ old: Double, _ new: Double) -> Double { old * 0.5 + new * 0.5 }
            let tokens = Double(max(0, current.decodeTokens - previous.state.decodeTokens)) / seconds
            let prefill = Double(max(0, current.prefillTokens - previous.state.prefillTokens)) / seconds
            let received = Double(max(0, current.bytesIn - previous.state.bytesIn)) / seconds
            let sent = Double(max(0, current.bytesOut - previous.state.bytesOut)) / seconds
            tokensPerSecond = smooth(tokensPerSecond, tokens)
            prefillPerSecond = smooth(prefillPerSecond, prefill)
            inPerSecond = smooth(inPerSecond, received)
            outPerSecond = smooth(outPerSecond, sent)
            append(&tokenHistory, current.mode.hasPrefix("LingBot") ? current.passMs : tokensPerSecond)
            append(&linkHistory, inPerSecond + outPerSecond)
        }
        previous = (current, now)
        state = current
        footprint = RemoteStageServer.footprint()
        available = Int(os_proc_available_memory())
        gpuActive = Memory.activeMemory
        thermal = ProcessInfo.processInfo.thermalState
    }

    private func append(_ history: inout [Double], _ value: Double) {
        history.append(value)
        if history.count > Self.historyLength { history.removeFirst(history.count - Self.historyLength) }
    }

    enum Activity {
        case off, waiting, loading, working, idle

        var label: String {
            switch self {
            case .off: "not serving"
            case .waiting: "waiting for a node"
            case .loading: "loading layers"
            case .working: "running"
            case .idle: "idle"
            }
        }

        var color: Color {
            switch self {
            case .off: .gray
            case .waiting: .secondary
            case .loading: .orange
            case .working: .green
            case .idle: .blue
            }
        }
    }

    var activity: Activity {
        if state.mode.isEmpty { return .off }
        if state.peer == nil { return .waiting }
        if state.layersLoaded < state.layersExpected { return .loading }
        if let last = state.lastForwardAt, Date().timeIntervalSince(last) < 1.5 { return .working }
        return .idle
    }

    var isSplit: Bool { state.mode.hasPrefix("LingBot") }
}

/// The panel beside the log: what's loaded, how fast it runs, what crosses the link, what the
/// caches hold
struct StatsPanel: View {
    @ObservedObject var model: StatsModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 12)], spacing: 12) {
                    inference
                    link
                    cache
                    device
                }
            }
            .padding(16)
        }
        .background(Color(.systemGroupedBackground))
    }

    // MARK: header

    private var header: some View {
        let state = model.state
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle()
                    .fill(model.activity.color)
                    .frame(width: 10, height: 10)
                    .shadow(color: model.activity.color.opacity(0.8), radius: model.activity == .working ? 4 : 0)
                Text(model.activity.label.uppercased())
                    .font(.caption.weight(.bold))
                    .foregroundStyle(model.activity.color)
                Spacer()
                if !state.mode.isEmpty {
                    Text("\(state.mode) · :\(String(state.port))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(state.modelName.isEmpty ? "No model yet" : state.modelName)
                .font(.system(.title, design: .rounded).weight(.bold))
                .lineLimit(2)
                .minimumScaleFactor(0.6)
            if !state.modelDetail.isEmpty {
                Text(state.modelDetail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            if state.layersExpected > 0 {
                if state.layersLoaded < state.layersExpected {
                    ProgressView(value: Double(state.layersLoaded), total: Double(state.layersExpected)) {
                        Text("Loading layer \(state.layersLoaded) of \(state.layersExpected)")
                            .font(.caption)
                    }
                    .tint(.orange)
                } else {
                    Text("\(state.layersExpected) \(model.isSplit ? "blocks" : "layers") here · "
                         + "\(Format.bytes(state.weightBytes)) of weights")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let peer = state.peer {
                Label {
                    if let since = state.connectedAt {
                        Text("\(peer) · connected ") + Text(since, style: .relative)
                    } else {
                        Text(peer)
                    }
                } icon: {
                    Image(systemName: "cable.connector")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: cards

    private var inference: some View {
        let state = model.state
        return Card(title: "Inference", symbol: "bolt.fill", tint: .green) {
            if model.isSplit {
                BigNumber(value: state.passes == 0 ? "–" : Format.seconds(state.passMs / 1000), unit: "per pass")
                Sparkline(values: model.tokenHistory, tint: .green)
                Row("Waiting on the Mac", state.passes == 0 ? "–" : Format.seconds(state.waitMs / 1000))
                Row("Tokens here", state.chunkTokens == 0 ? "–"
                    : "\(Format.count(state.shareTokens)) of \(Format.count(state.chunkTokens))")
                Row("Passes", Format.count(state.passes))
            } else {
                BigNumber(value: Format.rate(model.tokensPerSecond), unit: "tok/s")
                Sparkline(values: model.tokenHistory, tint: .green)
                Row("Forward on the GPU", state.forwards == 0 ? "–" : String(format: "%.1f ms", state.lastForwardMs))
                Row("Prefill", model.prefillPerSecond < 1 ? "–" : "\(Format.rate(model.prefillPerSecond)) tok/s")
                Row("Tokens", Format.count(state.decodeTokens + state.prefillTokens))
            }
        }
    }

    private var link: some View {
        let state = model.state
        return Card(title: "Link", symbol: "arrow.left.arrow.right", tint: .blue) {
            BigNumber(value: Format.bytes(Int(model.inPerSecond + model.outPerSecond), short: true), unit: "/s")
            Sparkline(values: model.linkHistory, tint: .blue)
            Row("↓ In", "\(Format.bytes(Int(model.inPerSecond)))/s")
            Row("↑ Out", "\(Format.bytes(Int(model.outPerSecond)))/s")
            Row("Moved", Format.bytes(state.bytesIn + state.bytesOut))
        }
    }

    private var cache: some View {
        let state = model.state
        return Card(title: "Cache", symbol: "square.stack.3d.up.fill", tint: .purple) {
            BigNumber(value: Format.bytes(state.cacheBytes, short: true), unit: "K/V")
            if model.isSplit {
                Row("Rows", Format.count(state.cacheRows))
            } else {
                Row("Sequences", Format.count(state.liveCaches))
                // A hybrid model's recurrent state counts here too, so this is an upper bound
                Row("Tokens held", state.kvBytesPerToken == 0 ? "–"
                    : "≈ \(Format.count(state.cacheBytes / state.kvBytesPerToken))")
                Row("Per token", state.kvBytesPerToken == 0 ? "–" : Format.bytes(state.kvBytesPerToken))
            }
        }
    }

    private var device: some View {
        let limit = model.footprint + model.available
        return Card(title: "iPad", symbol: "ipad.landscape", tint: .orange) {
            BigNumber(value: Format.bytes(model.footprint, short: true), unit: "in use")
            if limit > 0 {
                ProgressView(value: Double(model.footprint), total: Double(limit))
                    .tint(Double(model.footprint) / Double(limit) > 0.85 ? .red : .orange)
                Row("App limit", Format.bytes(limit))
            }
            Row("GPU arrays", Format.bytes(model.gpuActive))
            Row("Thermal", Format.thermal(model.thermal))
            Row("Keep-warm", model.state.mode.isEmpty ? "–" : model.state.keepWarm ? "on" : "off")
        }
    }
}

// MARK: pieces

private struct Card<Content: View>: View {
    let title: String
    let symbol: String
    let tint: Color
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label(title.uppercased(), systemImage: symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tint)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

private struct BigNumber: View {
    let value: String
    let unit: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(value)
                .font(.system(size: 32, weight: .bold, design: .rounded))
                .monospacedDigit()
                .contentTransition(.numericText())
            Text(unit)
                .font(.headline)
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .minimumScaleFactor(0.5)
    }
}

private struct Row: View {
    let label: String
    let value: String

    init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.subheadline)
    }
}

private struct Sparkline: View {
    let values: [Double]
    let tint: Color

    var body: some View {
        Chart(Array(values.enumerated()), id: \.offset) { index, value in
            AreaMark(x: .value("t", index), y: .value("v", value))
                .foregroundStyle(.linearGradient(
                    colors: [tint.opacity(0.35), tint.opacity(0.02)], startPoint: .top, endPoint: .bottom))
            LineMark(x: .value("t", index), y: .value("v", value))
                .foregroundStyle(tint)
                .lineStyle(StrokeStyle(lineWidth: 2))
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartXScale(domain: 0 ... max(StatsModel.historyLength - 1, values.count - 1))
        .chartYScale(domain: 0 ... max(values.max() ?? 1, 1) * 1.1)
        .frame(height: 34)
    }
}

enum Format {
    static func bytes(_ count: Int, short: Bool = false) -> String {
        let value = Double(count)
        // One decimal under 10 of a unit: 1.7 MB/s shouldn't read as 2
        func scaled(_ unit: Double, _ name: String) -> String {
            String(format: value / unit < 10 ? "%.1f %@" : "%.0f %@", value / unit, name)
        }
        if value >= 1e9 { return short ? scaled(1e9, "GB") : String(format: "%.2f GB", value / 1e9) }
        if value >= 1e6 { return scaled(1e6, "MB") }
        if value >= 1e3 { return scaled(1e3, "KB") }
        return "\(count) B"
    }

    static func rate(_ value: Double) -> String {
        value >= 1000 ? String(format: "%.1fk", value / 1000) : String(format: "%.0f", value)
    }

    static func count(_ value: Int) -> String {
        value.formatted(.number)
    }

    static func seconds(_ value: Double) -> String {
        value >= 1 ? String(format: "%.2f s", value) : String(format: "%.0f ms", value * 1000)
    }

    static func thermal(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "?"
        }
    }
}
