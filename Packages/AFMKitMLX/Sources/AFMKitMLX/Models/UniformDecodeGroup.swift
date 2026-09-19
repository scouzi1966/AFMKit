import Foundation
import MLX
import MLXLMCommon

/// Persistent, actor-owned decode storage for compatible rows of a mixed
/// cohort. Model adapters still decide whether their complete state is covered
/// by UniformBatchKVCache. This helper never pads or guesses recurrent state.
final class UniformDecodeGroup {
    private struct TensorSignature: Hashable {
        let shape: [Int]
        let dtype: String
    }

    private struct LayerSignature: Hashable {
        let type: ObjectIdentifier
        let offset: Int
        let metadata: [String]
        let tensors: [TensorSignature]
    }

    private(set) var slotIDs: [UUID]
    let caches: [KVCache]

    private static func signature(_ caches: [KVCache]) -> [LayerSignature]? {
        guard !caches.isEmpty else { return nil }
        var layers: [LayerSignature] = []
        for cache in caches {
            guard cache is UniformBatchKVCache else { return nil }
            let state = cache.state
            // Inputs must be populated request-owned rows, not an existing batch.
            guard !state.isEmpty,
                  state.allSatisfy({ $0.ndim > 0 && $0.dim(0) == 1 })
            else { return nil }
            layers.append(LayerSignature(
                type: ObjectIdentifier(type(of: cache)), offset: cache.offset,
                metadata: cache.metaState,
                tensors: state.map {
                    TensorSignature(shape: $0.shape, dtype: String(describing: $0.dtype))
                }))
        }
        return layers
    }

    /// Stable first-seen grouping, performed on admission, not every token.
    /// Incompatible/unknown cache types remain on the independent path.
    static func compatibleIndices(_ candidates: [[KVCache]]) -> [[Int]] {
        var positions: [[LayerSignature]: Int] = [:]
        var groups: [[Int]] = []
        for (index, caches) in candidates.enumerated() {
            guard let key = signature(caches) else { continue }
            if let position = positions[key] {
                groups[position].append(index)
            } else {
                positions[key] = groups.count
                groups.append([index])
            }
        }
        return groups.filter { $0.count > 1 }
    }

    init?(slotIDs: [UUID], requestCaches: [[KVCache]]) {
        guard slotIDs.count > 1, slotIDs.count == requestCaches.count,
              Set(slotIDs).count == slotIDs.count,
              let first = requestCaches.first,
              let key = Self.signature(first),
              requestCaches.dropFirst().allSatisfy({ Self.signature($0) == key })
        else { return nil }
        self.slotIDs = slotIDs
        self.caches = first.indices.map { layer in
            (first[layer] as! UniformBatchKVCache).mergedUniformBatch(
                requestCaches.map { $0[layer] })
        }
    }

    /// Remove only this group's row. A one-row survivor stays in its current
    /// cache representation; never restore the now-stale admission cache.
    func remove(_ id: UUID) {
        guard let index = slotIDs.firstIndex(of: id) else { return }
        let keep = slotIDs.indices.filter { $0 != index }
        if !keep.isEmpty {
            for cache in caches {
                (cache as! UniformBatchKVCache).filterUniformBatch(keep)
            }
        }
        slotIDs = keep.map { slotIDs[$0] }
    }
}
