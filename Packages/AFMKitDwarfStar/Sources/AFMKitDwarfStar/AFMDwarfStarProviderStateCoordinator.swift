import AFMKitCore
import Foundation
import os

/// Single canonical owner for DwarfStar admission and scheduler gauges.
final class AFMDwarfStarProviderStateCoordinator: @unchecked Sendable {
    static let shared = AFMDwarfStarProviderStateCoordinator()

    private struct State {
        var admissionWaiters: Set<UUID> = []
        var admissionReservations: Set<UUID> = []
        var schedulerRunning = 0
        var schedulerWaiting = 0
        var activeLogicalCachePositions = 0
        var logicalCacheCapacity = 0
        var observers: [UUID: any AFMInferenceTelemetryObserving] = [:]
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let publicationQueue = DispatchQueue(
        label: "com.maclocal.afmkit.dwarfstar.provider-state"
    )

    func register(_ observer: any AFMInferenceTelemetryObserving) -> UUID {
        let registration = UUID()
        state.withLock { state in
            state.observers[registration] = observer
            enqueue(
                snapshot: Self.snapshot(state),
                observers: [observer]
            )
        }
        return registration
    }

    func unregister(_ registration: UUID) {
        _ = state.withLock { $0.observers.removeValue(forKey: registration) }
    }

    func admissionStarted(_ id: UUID) {
        publishMutation { state in
            state.admissionWaiters.insert(id)
        }
    }

    func admissionReserved(_ id: UUID) {
        publishMutation { state in
            state.admissionWaiters.remove(id)
            state.admissionReservations.insert(id)
        }
    }

    func admissionFailed(_ id: UUID) {
        publishMutation { state in
            state.admissionWaiters.remove(id)
        }
    }

    func admissionReleased(_ id: UUID) {
        publishMutation { state in
            state.admissionReservations.remove(id)
        }
    }

    func schedulerChanged(
        running: Int,
        waiting: Int,
        activeLogicalCachePositions: Int,
        logicalCacheCapacity: Int
    ) {
        publishMutation { state in
            state.schedulerRunning = max(0, running)
            state.schedulerWaiting = max(0, waiting)
            state.activeLogicalCachePositions = max(0, activeLogicalCachePositions)
            state.logicalCacheCapacity = max(0, logicalCacheCapacity)
        }
    }

    private func publishMutation(_ mutation: @Sendable (inout State) -> Void) {
        state.withLock { state in
            mutation(&state)
            enqueue(
                snapshot: Self.snapshot(state),
                observers: Array(state.observers.values)
            )
        }
    }

    /// Enqueue while the state lock is held so delivery order exactly matches
    /// mutation order, but invoke observer code later on the serial queue.
    private func enqueue(
        snapshot: AFMInferenceProviderState,
        observers: [any AFMInferenceTelemetryObserving]
    ) {
        publicationQueue.async {
            for observer in observers {
                observer.updateProviderState(snapshot)
            }
        }
    }

    private static func snapshot(_ state: State) -> AFMInferenceProviderState {
        let handoffs = max(
            0,
            state.admissionReservations.count
                - state.schedulerRunning
                - state.schedulerWaiting
        )
        return AFMInferenceProviderState(
            runningRequests: state.schedulerRunning + handoffs,
            waitingRequests: state.schedulerWaiting + state.admissionWaiters.count,
            activeLogicalCachePositions: state.activeLogicalCachePositions,
            logicalCacheCapacity: state.logicalCacheCapacity
        )
    }
}
