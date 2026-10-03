import MLX
import MLXLMCommon

/// One caller-owned bank for ordinary mixed-position GDN decoding. Like the
/// MIT-licensed mlx-serve PersistentSsmGroup/ssmBindTickState (transformer.zig,
/// 1745ffe89e4670f1e0c6de22c75a9875b27399de), stable membership avoids per-tick
/// concatenation. Unlike that owner, we always publish current immutable row
/// views so existing request snapshots and singleton continuations stay valid.
/// PLE slots 2/3 and attention histories are never packed here.
final class Qwen4ExpRetainedBatchState: RequestOwnedDecodeBatchState {
    private struct Entry {
        let arrays: [MLXArray]
        let views: [[MLXArray]]
        let cacheIDs: [ObjectIdentifier]
        let bytes: Int
    }
    let modelID: ObjectIdentifier
    private var layers: [Int: Entry] = [:]
    private(set) var retainedBytes = 0
    private(set) var peakRetainedBytes = 0
    private(set) var layerHits = 0
    private(set) var layerRebuilds = 0

    init(model: AnyObject) { modelID = ObjectIdentifier(model) }

    func reset() {
        layers.removeAll(keepingCapacity: true)
        retainedBytes = 0
    }

    func restore(layer: Int, rows: [Qwen4ExpLayerCache]) -> [MLXArray]? {
        if let entry = layers[layer], entry.cacheIDs == rows.map(ObjectIdentifier.init),
           zip(rows, entry.views).allSatisfy({ row, views in
               (0..<2).allSatisfy { row[$0] === views[$0] }
           }) {
            layerHits += 1
            return entry.arrays
        }
        // Identity catches replacement/restore even when an adapter caller
        // misses a reset. In-place tensor writes still require caller reset,
        // as explicitly required by RequestOwnedDecodeBatchState's contract.
        if let prior = layers.removeValue(forKey: layer) { retainedBytes -= prior.bytes }
        layerRebuilds += 1
        return nil
    }

    func store(layer: Int, merged: Qwen4ExpLayerCache, rows: [Qwen4ExpLayerCache]) {
        let arrays = [merged[0]!, merged[1]!]
        let bytes = arrays.reduce(0) { $0 + $1.nbytes }
        let views = rows.map { [$0[0]!, $0[1]!] }
        if let prior = layers[layer] { retainedBytes -= prior.bytes }
        layers[layer] = Entry(arrays: arrays, views: views,
            cacheIDs: rows.map(ObjectIdentifier.init), bytes: bytes)
        retainedBytes += bytes
        peakRetainedBytes = max(peakRetainedBytes, retainedBytes)
    }
}
