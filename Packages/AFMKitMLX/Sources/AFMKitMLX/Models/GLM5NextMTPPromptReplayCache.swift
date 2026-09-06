import Foundation
import MLXLLM

/// Exact-prompt replay state for GLM's speculative path.
///
/// Unlike the generic radix tree, this entry also retains prompt hidden states
/// used to reseed the embedded NextN head. Access is locked because a future
/// scheduler may inspect it outside `ModelContainer.perform`.
final class GLM5NextMTPPromptReplayCache: @unchecked Sendable {
    private struct Entry {
        let tokens: [Int]
        let state: GLM5NextMTPGenerator.PromptState
        let retainedBytes: Int
    }

    private let modelID: String
    private let maxEntries: Int
    private let maxPromptTokens: Int
    private let maxRetainedBytes: Int
    private let lock = NSLock()
    private var entries: [Entry] = []
    private var retainedBytes = 0

    init(
        modelID: String,
        maxEntries: Int,
        maxPromptTokens: Int,
        maxRetainedBytes: Int
    ) {
        self.modelID = modelID
        self.maxEntries = max(1, maxEntries)
        self.maxPromptTokens = max(1, maxPromptTokens)
        self.maxRetainedBytes = max(1, maxRetainedBytes)
    }

    var currentRetainedBytes: Int {
        lock.withLock { retainedBytes }
    }

    func canStore(
        modelID: String,
        promptIds: [Int]
    ) -> Bool {
        modelID == self.modelID && promptIds.count <= maxPromptTokens
    }

    func findExactMatch(
        modelID: String,
        promptIds: [Int]
    ) -> GLM5NextMTPGenerator.PromptState? {
        lock.lock()
        defer { lock.unlock() }

        guard canStore(modelID: modelID, promptIds: promptIds) else {
            return nil
        }
        guard let entry = entries.last(where: { $0.tokens == promptIds }) else {
            return nil
        }
        return entry.state
    }

    func insert(
        _ state: GLM5NextMTPGenerator.PromptState,
        modelID: String
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard canStore(modelID: modelID, promptIds: state.promptIds) else {
            return false
        }

        if let existing = entries.firstIndex(where: { $0.tokens == state.promptIds }) {
            retainedBytes -= entries[existing].retainedBytes
            entries.remove(at: existing)
        }

        let entry = Entry(
            tokens: state.promptIds,
            state: state,
            retainedBytes: state.estimatedRetainedBytes)
        entries.append(entry)
        retainedBytes += entry.retainedBytes

        while retainedBytes > maxRetainedBytes || entries.count > maxEntries {
            if entries.count == 1 {
                retainedBytes -= entries.removeFirst().retainedBytes
                return false
            }
            retainedBytes -= entries.removeFirst().retainedBytes
        }
        return true
    }

    func invalidateAll() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
        retainedBytes = 0
    }

    var count: Int {
        lock.withLock { entries.count }
    }

    var usageFraction: Double {
        guard maxEntries > 0 else { return 0 }
        return Double(count) / Double(maxEntries)
    }
}
