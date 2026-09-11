/// Bounded exact-key storage for immutable, model-owned replay snapshots.
/// One model executor owns this cache; it adds no hot-path lock. Value adapters
/// must validate their model/generator identity and retain complete state.
final class ExactPromptReplayCache<Value> {
    private struct Entry {
        let prompt: [Int]
        let value: Value
        let bytes: Int
    }

    private let maximumBytes: Int
    private let maximumEntries: Int
    private let maximumPromptTokens: Int
    private var entries: [Entry] = []
    private(set) var retainedBytes = 0
    var count: Int { entries.count }

    init(maximumBytes: Int, maximumEntries: Int = 16, maximumPromptTokens: Int = 4096) {
        self.maximumBytes = max(0, maximumBytes)
        self.maximumEntries = max(0, maximumEntries)
        self.maximumPromptTokens = max(0, maximumPromptTokens)
    }

    func canStore(prompt: [Int]) -> Bool {
        maximumBytes > 0 && maximumEntries > 0
            && !prompt.isEmpty && prompt.count <= maximumPromptTokens
    }

    func find(prompt: [Int]) -> Value? {
        guard let index = entries.firstIndex(where: { $0.prompt == prompt }) else { return nil }
        let hit = entries.remove(at: index)
        entries.append(hit)
        return hit.value
    }

    @discardableResult
    func insert(prompt: [Int], value: Value, valueBytes: Int) -> Bool {
        guard canStore(prompt: prompt), valueBytes >= 0 else { return false }
        let (keyBytes, keyOverflow) = prompt.count.multipliedReportingOverflow(by: MemoryLayout<Int>.stride)
        let (bytes, overflow) = valueBytes.addingReportingOverflow(keyBytes)
        guard !keyOverflow && !overflow && bytes <= maximumBytes else { return false }
        if let index = entries.firstIndex(where: { $0.prompt == prompt }) {
            retainedBytes -= entries.remove(at: index).bytes
        }
        while !entries.isEmpty && (entries.count >= maximumEntries || retainedBytes > maximumBytes - bytes) {
            retainedBytes -= entries.removeFirst().bytes
        }
        entries.append(Entry(prompt: prompt, value: value, bytes: bytes))
        retainedBytes += bytes
        return true
    }

    func removeAll() {
        entries.removeAll()
        retainedBytes = 0
    }
}
