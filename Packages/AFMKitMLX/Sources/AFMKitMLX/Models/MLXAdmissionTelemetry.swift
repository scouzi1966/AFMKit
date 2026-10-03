import AFMKitCore
import Foundation

/// CPU-only bridge from admission state to the host's existing collector.
/// Lock order is service -> bridge -> individual counter locks. Counter
/// mutations must release their locks before calling `publish`. Readers must
/// not acquire the service lock or call GPU APIs. Observers must obey the core
/// synchronous, non-reentrant telemetry contract.
final class MLXAdmissionTelemetry: @unchecked Sendable {
    typealias Counts = (running: Int, waiting: Int)
    typealias Reader = @Sendable () -> Counts

    private let lock = NSLock()
    private let observer: any AFMInferenceTelemetryObserving
    private let serial: Reader
    private let admissionWaiters: @Sendable () -> Int
    private var batch: Reader?
    private var resources = AFMInferenceProviderState(runningRequests: 0, waitingRequests: 0)
    private var lastPublished: AFMInferenceProviderState?

    init(observer: any AFMInferenceTelemetryObserving,
         serial: @escaping Reader, admissionWaiters: @escaping @Sendable () -> Int) {
        self.observer = observer
        self.serial = serial
        self.admissionWaiters = admissionWaiters
    }

    /// Called under the service installation lock, including detachment.
    /// Notifications never carry snapshots or scheduler identities: a late
    /// notification from a retired scheduler re-reads ONLY the current source.
    func installBatchReader(_ reader: Reader?) {
        lock.withLock {
            batch = reader
            publishLocked(force: false)
        }
    }

    func publish(force: Bool = false) {
        lock.withLock { publishLocked(force: force) }
    }

    /// Resource sampling remains at the existing service admission boundary,
    /// never in scheduler membership transitions or the token loop.
    func updateResources(memoryCacheUsage: Double?, prefixCacheFill: Double?) {
        lock.withLock {
            resources.memoryCacheUsage = memoryCacheUsage
            resources.prefixCacheFill = prefixCacheFill
            publishLocked(force: false)
        }
    }

    private func publishLocked(force: Bool) {
        let counts = (batch ?? serial)()
        var state = resources
        state.runningRequests = max(0, counts.running)
        state.waitingRequests = max(0, counts.waiting) + max(0, admissionWaiters())
        guard force || state != lastPublished else { return }
        // Keep sampling AND delivery serialized. Releasing the lock between
        // them would let a delayed old snapshot overwrite a newer transition.
        observer.updateProviderState(state)
        lastPublished = state
    }
}
