import Foundation

/// Diagnostic host accounting, not isolated GPU kernel time. Graph construction
/// can overlap submitted GPU work; readout can wait for earlier lazy operations.
/// Construct only under AFM_PERF. No clock, eval or synchronization is owned here.
struct BatchExecutionProfile {
    enum Phase: String, CaseIterable {
        // Only prefillOne; speculative and dense batch prefills are not included.
        case prefillService = "ar-prefill-service"
        // MTP includes synchronous verification/repair here. Not CPU-only.
        case independentGraphSubmit = "independent-prepare-submit"
        case independentReadout = "independent-readout"
        case independentRetire = "independent-retire"
        case independentMaintenance = "independent-maintenance"
        // Whole tick, including cancellation cleanup and instrumentation gaps.
        // INCLUSIVE: never add this bucket to the component buckets above.
        case independentTotal = "independent-total-inclusive"
    }

    struct Sample: Equatable {
        var calls = 0
        var nanoseconds: UInt64 = 0
    }

    private(set) var samples: [Phase: [Int: Sample]] = [:]

    /// Each bucket is bounded by the scheduler's concurrency ceiling. Prefill
    /// rows are serviced requests. Whole-tick rows precede cancellation cleanup;
    /// component rows count survivors, not GPU width or tokens. Includes warmups.
    mutating func record(_ phase: Phase, rows: Int, nanoseconds: UInt64) {
        guard rows > 0 else { return }
        var sample = samples[phase]?[rows] ?? Sample()
        sample.calls += 1
        sample.nanoseconds += nanoseconds
        samples[phase, default: [:]][rows] = sample
    }

    var independentNanoseconds: UInt64 {
        samples[.independentTotal]?.values.reduce(0) { $0 + $1.nanoseconds } ?? 0
    }

    /// Remove nested independent decode from a prefill service span exactly once.
    /// Saturation tolerates an empty or coarse synthetic clock in unit tests.
    static func exclusiveNanoseconds(start: UInt64, end: UInt64, nested: UInt64) -> UInt64 {
        guard end >= start, end - start >= nested else { return 0 }
        return end - start - nested
    }

    var logLines: [String] {
        Phase.allCases.flatMap { phase in
            (samples[phase] ?? [:]).sorted { $0.key < $1.key }.map { rows, sample in
                "[BatchExecutionProfile] phase=\(phase.rawValue) "
                    + "\(phase == .prefillService ? "serviced_rows" : "active_rows")=\(rows) "
                    + "calls=\(sample.calls) host_ns=\(sample.nanoseconds)"
            }
        }
    }
}
