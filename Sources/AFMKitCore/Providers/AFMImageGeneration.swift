import Foundation

public struct AFMImageGenerationRequest: Hashable, Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var count: Int
    public var seed: UInt64?

    public init(prompt: String, width: Int, height: Int, count: Int = 1, seed: UInt64? = nil) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.count = count
        self.seed = seed
    }
}

public struct AFMImageEditRequest: Hashable, Sendable {
    public var prompt: String
    public var images: [Data]
    public var width: Int
    public var height: Int
    public var count: Int
    public var seed: UInt64?

    public init(
        prompt: String,
        images: [Data],
        width: Int,
        height: Int,
        count: Int = 1,
        seed: UInt64? = nil
    ) {
        self.prompt = prompt
        self.images = images
        self.width = width
        self.height = height
        self.count = count
        self.seed = seed
    }
}

public struct AFMGeneratedImage: Hashable, Sendable {
    public var data: Data
    public var mediaType: String
    public var width: Int
    public var height: Int

    public init(data: Data, mediaType: String = "image/png", width: Int, height: Int) {
        self.data = data
        self.mediaType = mediaType
        self.width = width
        self.height = height
    }
}

public protocol AFMImageGenerating: Sendable {
    func generateImages(for request: AFMImageGenerationRequest) async throws -> [AFMGeneratedImage]
    func editImages(for request: AFMImageEditRequest) async throws -> [AFMGeneratedImage]
}
