import Foundation
import MLXLMCommon

/// Serialized scheduler lifetime for one model's current AR batch. Packing and
/// geometry belong to the model adapter. Never accessed by cancellation tasks.
/// Disabled schedulers do not instantiate this owner or any model state.
final class RetainedRequestBatchOwner {
    let state: any RequestOwnedDecodeBatchState
    private var members: [UUID] = []
    private var cacheIDs: [[ObjectIdentifier]] = []
    private var suspensionDepth = 0
    private(set) var resets = 0

    init(state: any RequestOwnedDecodeBatchState) { self.state = state }

    func reset() {
        if !members.isEmpty || state.retainedBytes > 0 { resets += 1 }
        state.reset()
        members.removeAll(keepingCapacity: true)
        cacheIDs.removeAll(keepingCapacity: true)
    }

    func suspend() {
        reset()
        suspensionDepth += 1
    }

    func resume() {
        precondition(suspensionDepth > 0)
        reset()
        suspensionDepth -= 1
    }

    func prepare(rowIDs: [UUID], caches: [[KVCache]]) -> (any RequestOwnedDecodeBatchState)? {
        guard suspensionDepth == 0, rowIDs.count > 1, caches.count == rowIDs.count,
              Set(rowIDs).count == rowIDs.count, caches.allSatisfy({ !$0.isEmpty }) else {
            reset()
            return nil
        }
        let identities = caches.map { $0.map { ObjectIdentifier($0 as AnyObject) } }
        if rowIDs != members || identities != cacheIDs {
            reset()
            members = rowIDs
            cacheIDs = identities
        }
        return state
    }
}
