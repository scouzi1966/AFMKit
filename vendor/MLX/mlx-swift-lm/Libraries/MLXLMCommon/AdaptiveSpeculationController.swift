// Copyright © 2026 AFMKit contributors.
import Foundation

/// Request-local, bounded online selection of speculative work. The controller
/// changes proposal length, never the target sampler or acceptance rule.
/// Cost must include the observed draft/verify/commit interval, not just a
/// kernel-build timer. No process-global history or random state is retained.
public struct AdaptiveSpeculationController {
    public static let maximumSupportedDepth = 8
    private static let initialSamples = 2
    private static let probeInterval = 16
    private static let smoothing = 0.25
    private static let improvementMargin = 1.05
    private struct Arm {
        var samples = 0
        var tokens = 0.0
        var seconds = 0.0
        var score: Double { seconds > 0 ? tokens / seconds : 0 }
    }
    private var arms: [Arm]
    private var observations = 0
    private var probe = 0
    public private(set) var selectedDepth: Int

    public init(maximumDepth: Int) {
        selectedDepth = min(Self.maximumSupportedDepth, max(1, maximumDepth))
        arms = Array(repeating: Arm(), count: selectedDepth)
    }

    public mutating func observe(drafted: Int, accepted: Int, elapsedSeconds: Double) {
        guard (1...arms.count).contains(drafted), (0...drafted).contains(accepted),
              elapsedSeconds.isFinite, elapsedSeconds > 0 else { return }
        let i = drafted - 1
        let weight = arms[i].samples == 0 ? 1 : Self.smoothing
        arms[i].tokens += weight * (Double(accepted + 1) - arms[i].tokens)
        arms[i].seconds += weight * (elapsedSeconds - arms[i].seconds)
        arms[i].samples += 1
        observations += 1
        // Try each depth, starting at the requested maximum. Periodic bounded
        // probes allow recovery when the prompt changes from prose to code.
        if let untried = arms.indices.reversed().first(where: { arms[$0].samples < Self.initialSamples }) {
            selectedDepth = untried + 1
        } else if observations.isMultiple(of: Self.probeInterval) {
            selectedDepth = probe % arms.count + 1
            probe += 1
        } else if let best = arms.indices.max(by: { arms[$0].score < arms[$1].score }),
                  arms[best].score > arms[selectedDepth - 1].score * Self.improvementMargin {
            selectedDepth = best + 1
        }
    }
}

/// Owner-local aggregate-work experiment. Every ready request in a width band
/// uses one depth for a stable epoch. Observations span the entire owner step,
/// including deferred repair submissions and natural GPU waits, not per-row
/// draft/verify timers. No synchronization is added solely for measurement.
/// GPU work crossing an epoch boundary is amortized, not precisely attributed.
public struct CohortSpeculationController {
    public static let measurementSteps = 32
    private static let settlingSteps = AdaptiveSpeculationController.maximumSupportedDepth + 1
    private static let probeInterval = 8
    private static let improvementMargin = 1.05
    private struct Band {
        var selected: Int
        var settle = CohortSpeculationController.settlingSteps
        var steps = 0
        var tokens = 0
        var seconds = 0.0
        var scores: [Double?]
        var epochs = 0
    }
    private let maximumDepth: Int
    private var bands: [Int: Band] = [:]
    public private(set) var completedEpochs = 0
    public private(set) var depthChanges = 0

    public init(maximumDepth: Int) {
        self.maximumDepth = min(AdaptiveSpeculationController.maximumSupportedDepth, max(1, maximumDepth))
    }

    private func band(for activeRows: Int) -> Int {
        if activeRows <= 1 { return 1 }
        if activeRows <= 4 { return 4 }
        if activeRows <= 8 { return 8 }
        if activeRows <= 16 { return 16 }
        return 32
    }

    public func depth(activeRows: Int) -> Int {
        bands[band(for: activeRows)]?.selected ?? maximumDepth
    }

    public mutating func observe(activeRows: Int, emittedTokens: Int, elapsedSeconds: Double) {
        guard activeRows > 0, (0...activeRows).contains(emittedTokens),
              elapsedSeconds.isFinite, elapsedSeconds > 0 else { return }
        let key = band(for: activeRows)
        var state = bands[key] ?? Band(selected: maximumDepth,
            scores: Array(repeating: nil, count: maximumDepth))
        defer { bands[key] = state }
        if state.settle > 0 { state.settle -= 1; return }
        state.steps += 1
        state.tokens += emittedTokens
        state.seconds += elapsedSeconds
        guard state.steps >= Self.measurementSteps else { return }
        let score = Double(state.tokens) / state.seconds
        let previous = state.scores[state.selected - 1]
        state.scores[state.selected - 1] = previous.map { ($0 + score) / 2 } ?? score
        state.steps = 0
        state.tokens = 0
        state.seconds = 0
        state.epochs += 1
        completedEpochs += 1
        let next: Int
        if let untried = state.scores.indices.reversed().first(where: { state.scores[$0] == nil }) {
            next = untried + 1
        } else if state.epochs.isMultiple(of: Self.probeInterval) {
            next = (state.epochs / Self.probeInterval - 1) % maximumDepth + 1
        } else {
            let best = state.scores.indices.max { state.scores[$0]! < state.scores[$1]! }!
            next = state.scores[best]! > state.scores[state.selected - 1]! * Self.improvementMargin
                ? best + 1 : state.selected
        }
        if next != state.selected {
            state.selected = next
            state.settle = Self.settlingSteps
            depthChanges += 1
        }
    }
}
