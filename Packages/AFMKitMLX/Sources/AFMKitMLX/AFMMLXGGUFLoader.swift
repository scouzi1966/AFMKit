import Foundation
import MLX

/// A lightweight summary returned after MLX has parsed a GGUF checkpoint.
///
/// This POC deliberately exposes loading separately from model construction:
/// GGUF stores tensors and metadata, but not an executable computation graph.
public struct AFMMLXGGUFCheckpoint: Sendable {
    public let architecture: String?
    public let tensorCount: Int
    public let tensorNames: [String]
    public let totalElementCount: Int
    public let metadataKeys: [String]
}

public enum AFMMLXGGUFLoader {
    /// Parses and loads a GGUF checkpoint with MLX's native GGUF implementation.
    ///
    /// Supported packed quantizations currently follow upstream MLX: Q4_0,
    /// Q4_1, and Q8_0. Other recognized types may be expanded to float16.
    public static func load(url: URL) throws -> AFMMLXGGUFCheckpoint {
        let result = try MLX.loadGGUF(url: url)
        let architecture: String?
        if case .string(let value)? = result.metadata["general.architecture"] {
            architecture = value
        } else {
            architecture = nil
        }

        return AFMMLXGGUFCheckpoint(
            architecture: architecture,
            tensorCount: result.arrays.count,
            tensorNames: result.arrays.keys.sorted(),
            totalElementCount: result.arrays.values.reduce(0) { $0 + $1.size },
            metadataKeys: result.metadata.keys.sorted()
        )
    }
}
