import Foundation

/// Built-in checkpoint conversion dispatcher shared by AFM applications.
/// Provider-specific conversion remains in AFMKit; consumers only select a
/// local source, destination, and stable conversion profile.
public struct AFMMLXCheckpointConverter {
    public typealias ProgressHandler = (String) -> Void

    public enum ModelKind: String, Sendable, Equatable {
        case deepseekV4 = "deepseek_v4"
        case glm5Next = "glm5_next"
    }

    public struct Inspection: Sendable, Equatable {
        public let modelKind: ModelKind
        public let defaultProfile: String
        public let supportedProfiles: [String]
        public let sourceRevision: String?
        public let sourceBytes: Int64
        public let estimatedOutputBytes: Int64?
        public let requiredDestinationFreeBytes: Int64?

        public init(
            modelKind: ModelKind,
            defaultProfile: String,
            supportedProfiles: [String],
            sourceRevision: String?,
            sourceBytes: Int64,
            estimatedOutputBytes: Int64?,
            requiredDestinationFreeBytes: Int64?
        ) {
            self.modelKind = modelKind
            self.defaultProfile = defaultProfile
            self.supportedProfiles = supportedProfiles
            self.sourceRevision = sourceRevision
            self.sourceBytes = sourceBytes
            self.estimatedOutputBytes = estimatedOutputBytes
            self.requiredDestinationFreeBytes = requiredDestinationFreeBytes
        }
    }

    public struct ResumeInspection: Sendable, Equatable {
        public let sourceRevision: String?
        public let verifiedCompletedOutputBytes: Int64

        public init(
            sourceRevision: String?,
            verifiedCompletedOutputBytes: Int64
        ) {
            self.sourceRevision = sourceRevision
            self.verifiedCompletedOutputBytes = verifiedCompletedOutputBytes
        }
    }

    public enum ConversionError: LocalizedError, Equatable {
        case invalidSource(String)
        case unsupportedProfile(String)

        public var errorDescription: String? {
            switch self {
            case .invalidSource(let message), .unsupportedProfile(let message): message
            }
        }
    }

    let source: URL
    let output: URL
    let overwrite: Bool
    let profile: String?
    let sourceRevision: String?
    let progress: ProgressHandler?

    public init(
        source: URL,
        output: URL,
        overwrite: Bool = false,
        profile: String? = nil,
        sourceRevision: String? = nil,
        progress: ProgressHandler? = nil
    ) {
        self.source = source
        self.output = output
        self.overwrite = overwrite
        self.profile = profile
        self.sourceRevision = sourceRevision
        self.progress = progress
    }

    public static func inspect(
        source: URL,
        sourceRevision: String? = nil
    ) throws -> Inspection {
        let configURL = source.standardizedFileURL.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path),
              let object = try JSONSerialization.jsonObject(
                with: Data(contentsOf: configURL)) as? [String: Any],
              let modelType = object["model_type"] as? String
        else {
            throw ConversionError.invalidSource(
                "Local source directory must contain a valid config.json.")
        }
        switch modelType {
        case ModelKind.deepseekV4.rawValue:
            guard sourceRevision == nil else {
                throw ConversionError.invalidSource(
                    "--source-revision is supported only for GLM-5.3 conversion; DeepSeek conversion does not consume Hugging Face revision provenance.")
            }
            let profiles = DeepseekV4CheckpointConverter.Profile.allCases.map(\.rawValue)
            return Inspection(
                modelKind: .deepseekV4,
                defaultProfile: DeepseekV4CheckpointConverter.Profile.native.rawValue,
                supportedProfiles: profiles,
                sourceRevision: nil,
                sourceBytes: try checkpointBytes(source),
                estimatedOutputBytes: nil,
                requiredDestinationFreeBytes: nil)
        case ModelKind.glm5Next.rawValue:
            let inspection = try GLM5NextCheckpointConverter.inspect(
                source: source, sourceRevision: sourceRevision)
            return Inspection(
                modelKind: .glm5Next,
                defaultProfile: GLM5NextCheckpointConverter.Profile.mlxAffine4.rawValue,
                supportedProfiles: GLM5NextCheckpointConverter.Profile.allCases.map(\.rawValue),
                sourceRevision: inspection.sourceRevision,
                sourceBytes: inspection.sourceBytes,
                estimatedOutputBytes: inspection.estimatedOutputBytes,
                requiredDestinationFreeBytes: inspection.requiredDestinationFreeBytes)
        default:
            throw ConversionError.invalidSource(
                "Built-in conversion does not support model_type \(modelType).")
        }
    }

    /// Returns provider-verified resumable output bytes. Consumers must not
    /// decode provider-private conversion manifests themselves.
    public static func inspectResume(
        source: URL,
        output: URL,
        profile: String? = nil,
        sourceRevision: String? = nil
    ) throws -> ResumeInspection {
        let inspection = try inspect(source: source, sourceRevision: sourceRevision)
        let selected = profile ?? inspection.defaultProfile
        guard inspection.supportedProfiles.contains(selected) else {
            throw ConversionError.unsupportedProfile(
                "Profile '\(selected)' is not supported for \(inspection.modelKind.rawValue). Expected: \(inspection.supportedProfiles.joined(separator: ", ")).")
        }
        switch inspection.modelKind {
        case .deepseekV4:
            return ResumeInspection(
                sourceRevision: nil, verifiedCompletedOutputBytes: 0)
        case .glm5Next:
            let status = try GLM5NextCheckpointConverter.inspectResume(
                source: source,
                output: output,
                profile: GLM5NextCheckpointConverter.Profile(rawValue: selected)!,
                sourceRevision: inspection.sourceRevision)
            return ResumeInspection(
                sourceRevision: status.sourceRevision,
                verifiedCompletedOutputBytes: status.verifiedCompletedOutputBytes)
        }
    }

    public func run() throws {
        let inspection = try Self.inspect(
            source: source, sourceRevision: sourceRevision)
        let selected = profile ?? inspection.defaultProfile
        guard inspection.supportedProfiles.contains(selected) else {
            throw ConversionError.unsupportedProfile(
                "Profile '\(selected)' is not supported for \(inspection.modelKind.rawValue). Expected: \(inspection.supportedProfiles.joined(separator: ", ")).")
        }
        switch inspection.modelKind {
        case .deepseekV4:
            let deepseekProfile = DeepseekV4CheckpointConverter.Profile(rawValue: selected)!
            try DeepseekV4CheckpointConverter(
                source: source,
                output: output,
                overwrite: overwrite,
                profile: deepseekProfile,
                progress: progress).run()
        case .glm5Next:
            let glmProfile = GLM5NextCheckpointConverter.Profile(rawValue: selected)!
            try GLM5NextCheckpointConverter(
                source: source,
                output: output,
                overwrite: overwrite,
                profile: glmProfile,
                sourceRevision: inspection.sourceRevision,
                progress: progress).run()
        }
    }

    private static func checkpointBytes(_ source: URL) throws -> Int64 {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])
        return try files.filter { $0.pathExtension == "safetensors" }.reduce(Int64(0)) {
            $0 + Int64(try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        }
    }
}
