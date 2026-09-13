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
