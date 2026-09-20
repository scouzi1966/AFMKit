/// Bounded exact-key storage for immutable, model-owned replay snapshots.
/// One model executor owns this cache; it adds no hot-path lock. Value adapters
/// must validate their model/generator identity and retain complete state.
final class ExactPromptReplayCache<Value> {
    private struct Entry {
        let prompt: [Int]
        let value: Value
        let bytes: Int
        let sourcePrompt: [Int]?
        var sharedReuse = false

        var isEarlierBoundary: Bool {
            guard let sourcePrompt else { return false }
            return !prompt.isEmpty && prompt.count < sourcePrompt.count
                && sourcePrompt.starts(with: prompt)
        }
    }

    private let maximumBytes: Int
    private let maximumEntries: Int
    private let preserveCoverageAnchor: Bool
    let maximumPromptTokens: Int
    private var entries: [Entry] = []
    private(set) var retainedBytes = 0
    var count: Int { entries.count }

    init(maximumBytes: Int, maximumEntries: Int = 16, maximumPromptTokens: Int = 4096,
         preserveCoverageAnchor: Bool = false) {
        self.maximumBytes = max(0, maximumBytes)
        self.maximumEntries = max(0, maximumEntries)
        self.maximumPromptTokens = max(0, maximumPromptTokens)
        self.preserveCoverageAnchor = preserveCoverageAnchor
    }

    func canStore(prompt: [Int]) -> Bool {
        maximumBytes > 0 && maximumEntries > 0
            && !prompt.isEmpty && prompt.count <= maximumPromptTokens
    }

    /// Prefix matching is opt-in: only an adapter that can continue its complete
    /// saved state may request it. Pick the longest saved boundary, never trim
    /// recurrence back from a longer or merely overlapping prompt.
    func find(prompt: [Int], allowPrefix: Bool = false) -> Value? {
        var match: Int?
        for index in entries.indices {
            let key = entries[index].prompt
            let matches = allowPrefix ? prompt.starts(with: key) : prompt == key
            if matches, match == nil || key.count > entries[match!].prompt.count {
                match = index
            }
        }
        guard let index = match else { return nil }
        var hit = entries.remove(at: index)
        if let source = hit.sourcePrompt, source != prompt {
            hit.sharedReuse = true
        }
        entries.append(hit)
        return hit.value
    }

    @discardableResult
    func insert(prompt: [Int], value: Value, valueBytes: Int, sourcePrompt: [Int]? = nil) -> Bool {
        guard canStore(prompt: prompt), valueBytes >= 0 else { return false }
        // Opt-in ownership metadata, never inference input. An exact repeat
        // promotes its unshared earlier snapshot in place, not into a second
        // LRU entry. Keep a boundary that has actually served another prompt.
        // Metadata is bounded/accounted; default callers keep exact-key policy.
        if let sourcePrompt {
            guard canStore(prompt: sourcePrompt), sourcePrompt.starts(with: prompt) else { return false }
        }
        let (keyBytes, keyOverflow) = prompt.count.multipliedReportingOverflow(by: MemoryLayout<Int>.stride)
        let (sourceBytes, sourceOverflow) = (sourcePrompt?.count ?? 0).multipliedReportingOverflow(by: MemoryLayout<Int>.stride)
        let (keyAndSourceBytes, metadataOverflow) = keyBytes.addingReportingOverflow(sourceBytes)
        let (bytes, overflow) = valueBytes.addingReportingOverflow(keyAndSourceBytes)
        guard !keyOverflow && !sourceOverflow && !metadataOverflow && !overflow && bytes <= maximumBytes else { return false }
        // Experimental retention only: choose before promotion deletes an
        // unused donor. Distinct known source prompts are evidence of possible
        // reuse even if an endpoint has so far won every actual lookup.
        // Freeze the key, not an index that changes during removals. No new
        // snapshot, numerical boundary or model state is constructed here.
        let anchor = coverageAnchor(adding: sourcePrompt)
        var sharedReuse = false
        for index in entries.indices.reversed() {
            let entry = entries[index]
            let exactKey = entry.prompt == prompt
            let unsharedPromotion = sourcePrompt != nil && entry.sourcePrompt == sourcePrompt
                && !entry.sharedReuse && prompt.starts(with: entry.prompt)
                && entry.prompt != anchor
            if exactKey || unsharedPromotion {
                if exactKey { sharedReuse = entry.sharedReuse }
                retainedBytes -= entries.remove(at: index).bytes
            }
        }
        while !entries.isEmpty && (entries.count >= maximumEntries || retainedBytes > maximumBytes - bytes) {
            // Under pressure prefer another earlier snapshot to an endpoint.
            // With no useful anchor this is exactly the original LRU policy.
            // Protection is best effort: a fitting new entry always respects
            // the existing byte/entry caps, even if the anchor must be evicted.
            let victim: Int
            if let anchor {
                victim = entries.firstIndex { $0.isEarlierBoundary && $0.prompt != anchor }
                    ?? entries.firstIndex { $0.prompt != anchor } ?? 0
            } else {
                victim = 0
            }
            retainedBytes -= entries.remove(at: victim).bytes
        }
        entries.append(Entry(prompt: prompt, value: value, bytes: bytes,
            sourcePrompt: sourcePrompt, sharedReuse: sharedReuse))
        retainedBytes += bytes
        return true
    }

    /// At most one best-effort anchor. This runs only on insertion, never
    /// decode/lookup, and is bounded by the existing small entry/token caps.
    /// No historical popularity counter or model-specific token ID is used.
    /// Coverage ties prefer the deeper reusable state, then the newer entry.
    private func coverageAnchor(adding sourcePrompt: [Int]?) -> [Int]? {
        guard preserveCoverageAnchor else { return nil }
        var sources = Set(entries.compactMap(\.sourcePrompt))
        if let sourcePrompt { sources.insert(sourcePrompt) }
        var selected: [Int]?
        var coverage = 0
        for entry in entries where entry.isEarlierBoundary {
            let count = sources.reduce(0) { count, source in
                count + (source != entry.sourcePrompt && source.starts(with: entry.prompt) ? 1 : 0)
            }
            if count > coverage || (count > 0 && count == coverage
                && entry.prompt.count >= (selected?.count ?? 0)) {
                selected = entry.prompt
                coverage = count
            }
        }
        return selected
    }

    func removeAll() {
        entries.removeAll()
        retainedBytes = 0
    }
}
