import AFMKitCore
import CoreGraphics
import Foundation
import ImageIO
import MLX
import Tokenizers
import UniformTypeIdentifiers

public enum FluxKleinImageError: Error, LocalizedError, Sendable {
    case unreadableSnapshot(String)
    case invalidImage
    case pngEncodingFailed

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let path):
            return "FLUX.2-klein snapshot is not readable at \(path)."
        case .invalidImage:
            return "The input image could not be decoded."
        case .pngEncodingFailed:
            return "The generated image could not be encoded as PNG."
        }
    }
}

public enum FluxKleinQuantization: String, Hashable, Sendable {
    case bf16
    case int8
    case int4
}

public struct FluxKleinImageConfiguration: Hashable, Sendable {
    public var modelID: String
    public var cacheDirectory: URL
    public var quantization: FluxKleinQuantization
    public var steps: Int

    public init(
        modelID: String = "mlx-community/FLUX.2-klein-4B-bf16",
        cacheDirectory: URL,
        quantization: FluxKleinQuantization = .int4,
        steps: Int = 4
    ) {
        self.modelID = modelID
        self.cacheDirectory = cacheDirectory
        self.quantization = quantization
        self.steps = steps
    }

    public var snapshotDirectory: URL {
        cacheDirectory.appendingPathComponent(modelID, isDirectory: true)
    }
}

/// Lazy, serialized FLUX.2-klein image generation backed by AFMKit's single MLX runtime.
///
/// The model repository stores bf16 weights; `int4` and `int8` quantize the transformer
/// after loading so the same downloaded snapshot serves every memory tier.
public actor FluxKleinImageService: AFMImageGenerating {
    private let configuration: FluxKleinImageConfiguration
    private var generator: KleinGenerator?

    public init(configuration: FluxKleinImageConfiguration) {
        self.configuration = configuration
    }

    public func unload() {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func generateImages(for request: AFMImageGenerationRequest) async throws -> [AFMGeneratedImage] {
        try Task.checkCancellation()
        let generator = try await loadedGenerator()
        var images: [AFMGeneratedImage] = []
        images.reserveCapacity(request.count)
        for index in 0..<request.count {
            try Task.checkCancellation()
            let seed = (request.seed ?? UInt64.random(in: UInt64.min...UInt64.max)) &+ UInt64(index)
            let output = generator.generate(
                prompt: request.prompt,
                width: aligned(request.width),
                height: aligned(request.height),
                steps: configuration.steps,
                seed: seed
            )
            try Task.checkCancellation()
            images.append(AFMGeneratedImage(
                data: try Self.encodePNG(pixels: output.pixels, width: output.width, height: output.height),
                width: output.width,
                height: output.height
            ))
        }
        return images
    }

    public func editImages(for request: AFMImageEditRequest) async throws -> [AFMGeneratedImage] {
        try Task.checkCancellation()
        let generator = try await loadedGenerator()
        let width = aligned(request.width)
        let height = aligned(request.height)
        let references = try request.images.map { try Self.decodeReference($0, width: width, height: height) }
        var images: [AFMGeneratedImage] = []
        images.reserveCapacity(request.count)
        for index in 0..<request.count {
            try Task.checkCancellation()
            let seed = (request.seed ?? UInt64.random(in: UInt64.min...UInt64.max)) &+ UInt64(index)
            let output = generator.generateEdit(
                prompt: request.prompt,
                referenceImages: references,
                width: width,
                height: height,
                steps: configuration.steps,
                seed: seed
            )
            try Task.checkCancellation()
            images.append(AFMGeneratedImage(
                data: try Self.encodePNG(pixels: output.pixels, width: output.width, height: output.height),
                width: output.width,
                height: output.height
            ))
        }
        return images
    }

    private func aligned(_ dimension: Int) -> Int {
        max(64, (dimension / 16) * 16)
    }

    private func loadedGenerator() async throws -> KleinGenerator {
        if let generator { return generator }
        let snapshot = configuration.snapshotDirectory
        guard FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("text_encoder").path),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("vae").path)
        else {
            throw FluxKleinImageError.unreadableSnapshot(snapshot.path)
        }

        let transformer = try KleinWeights.loadTransformer(snapshotPath: snapshot.path, dtype: .bfloat16)
        switch configuration.quantization {
        case .bf16:
            break
        case .int8:
            KleinWeights.quantizeDiT(transformer, bits: 8)
        case .int4:
            KleinWeights.quantizeDiT(transformer, bits: 4)
        }
        let vae = try Flux2VAEWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae"),
            dtype: .float32
        )
        let vaeEncoder = try KleinWeights.loadVAEEncoder(snapshotPath: snapshot.path, dtype: .float32)
        let vaeArrays = try MLX.loadArrays(
            url: snapshot.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        )
        guard let runningMean = vaeArrays["bn.running_mean"],
              let runningVariance = vaeArrays["bn.running_var"]
        else {
            throw FluxKleinImageError.unreadableSnapshot(snapshot.path)
        }
        let bnMean = runningMean.asType(.float32).reshaped(1, -1, 1, 1)
        let bnStd = MLX.sqrt(runningVariance.asType(.float32).reshaped(1, -1, 1, 1) + 1e-4)
        eval(bnMean, bnStd)

        let encoder = try KleinWeights.loadTextEncoder(snapshotPath: snapshot.path, dtype: .bfloat16)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer", isDirectory: true)
        )
        let loaded = KleinGenerator(
            transformer: transformer,
            vae: vae,
            textEncoder: KleinTextEncoder(encoder: encoder, tokenizer: tokenizer),
            transformerDtype: .bfloat16,
            vaeEncoder: vaeEncoder,
            bnMean: bnMean,
            bnStd: bnStd
        )
        generator = loaded
        return loaded
    }

    private nonisolated static func decodeReference(_ data: Data, width: Int, height: Int) throws -> MLXArray {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        else {
            throw FluxKleinImageError.invalidImage
        }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw FluxKleinImageError.invalidImage
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [Float](repeating: 0, count: 3 * width * height)
        let plane = width * height
        for pixel in 0..<plane {
            rgb[pixel] = Float(bytes[pixel * 4]) / 127.5 - 1
            rgb[plane + pixel] = Float(bytes[pixel * 4 + 1]) / 127.5 - 1
            rgb[2 * plane + pixel] = Float(bytes[pixel * 4 + 2]) / 127.5 - 1
        }
        return MLXArray(rgb, [1, 3, height, width])
    }

    private nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        guard pixels.count == width * height * 3,
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              let buffer = context.data
        else {
            throw FluxKleinImageError.pngEncodingFailed
        }
        let rgba = buffer.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for pixel in 0..<(width * height) {
            rgba[pixel * 4] = pixels[pixel * 3]
            rgba[pixel * 4 + 1] = pixels[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = pixels[pixel * 3 + 2]
            rgba[pixel * 4 + 3] = 255
        }
        guard let image = context.makeImage() else {
            throw FluxKleinImageError.pngEncodingFailed
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw FluxKleinImageError.pngEncodingFailed
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FluxKleinImageError.pngEncodingFailed
        }
        return data as Data
    }
}
