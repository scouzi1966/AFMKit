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
    }

    private let modelID: String
    private let maxEntries: Int
    private let lock = NSLock()
    private var entries: [Entry] = []

    init(modelID: String, maxEntries: Int) {
        self.modelID = modelID
        self.maxEntries = max(1, maxEntries)
    }

    func findExactMatch(
        modelID: String,
        promptIds: [Int]
    ) -> GLM5NextMTPGenerator.PromptState? {
        lock.lock()
        defer { lock.unlock() }

        guard modelID == self.modelID else { return nil }
        guard let entry = entries.last(where: { $0.tokens == promptIds }) else {
            return nil
        }
        return entry.state
    }

    func insert(
        _ state: GLM5NextMTPGenerator.PromptState,
        modelID: String
    ) {
        lock.lock()
        defer { lock.unlock() }

        guard modelID == self.modelID else { return }
        entries.removeAll { $0.tokens == state.promptIds }
        entries.append(Entry(tokens: state.promptIds, state: state))
        while entries.count > maxEntries {
            entries.removeFirst()
        }
    }

    func invalidateAll() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
    }

    var count: Int {
        lock.withLock { entries.count }
    }

    var usageFraction: Double {
        guard maxEntries > 0 else { return 0 }
        return Double(count) / Double(maxEntries)
    }
}
