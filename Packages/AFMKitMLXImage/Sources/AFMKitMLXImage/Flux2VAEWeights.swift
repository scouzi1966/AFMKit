// FLUX.2 VAE weight loading — Swift/MLX. Decoder-only: encoder.* / quant_conv.* keys are skipped.
//
// diffusers-identical keys except `to_out.0.` -> `to_out.`, drop `num_batches_tracked`, and 4D
// conv weights transpose PT (O,I,kH,kW) -> MLX (O,kH,kW,I). The `bn` running stats stay fp32
// regardless of requested dtype (de-norm precision). Two-way strict load (workspace discipline):
// every module key must be filled and every checkpoint key consumed — a partial load emits
// garbage with no other symptom.

import Foundation
import MLX
import MLXNN

public enum Flux2VAEError: Error, CustomStringConvertible {
    case loading(String)
    public var description: String {
        switch self { case .loading(let m): return "Flux2VAE load: \(m)" }
    }
}

public enum Flux2VAEWeights {

    /// Load the FLUX.2 VAE (decoder-only) from a diffusers `vae/` snapshot.
    public static func loadVAE(directory: URL, dtype: DType = .float32) throws -> Flux2VAE {
        let vae = Flux2VAE()
        var state: [String: MLXArray] = [:]
        for (rawKey, rawValue) in try loadAllArrays(directory: directory) {
            if rawKey.hasSuffix("num_batches_tracked") { continue }
            // Decoder-only: skip the encoder tower + its quant projection.
            if rawKey.hasPrefix("encoder.") || rawKey.hasPrefix("quant_conv.") { continue }
            let k = rawKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            var v = rawValue
            if v.ndim == 4 {  // conv weight: PT (O,I,kH,kW) -> MLX (O,kH,kW,I)
                v = v.transposed(0, 2, 3, 1)
            }
            // bn stats stay fp32 regardless of requested dtype (de-norm precision).
            v = k.hasPrefix("bn.") ? v.asType(.float32) : v.asType(dtype)
            state[k] = v
        }
        try verifyAndLoad(model: vae, weights: state, label: "VAE")
        return vae
    }

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw Flux2VAEError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    /// Two-way strict load: all module keys must be filled and every provided key consumed.
    static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw Flux2VAEError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw Flux2VAEError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}
