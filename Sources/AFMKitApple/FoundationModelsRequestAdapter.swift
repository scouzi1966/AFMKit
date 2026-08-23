#if canImport(FoundationModels)
import AFMKitCore
import Darwin
import Foundation
import FoundationModels
import ImageIO
import UniformTypeIdentifiers

struct AFMRuntimeSymbolResolver: Sendable {
    private let isPresent: @Sendable (String) -> Bool

    init(isPresent: @escaping @Sendable (String) -> Bool) {
        self.isPresent = isPresent
    }

    func contains(_ symbol: String) -> Bool {
        isPresent(symbol)
    }

    static let process = AFMRuntimeSymbolResolver { symbol in
        dlsym(UnsafeMutableRawPointer(bitPattern: -2), symbol) != nil
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationModelsRequestAdapter {
    public static func request<Model: AFMFoundationModelsModelConfiguration>(
        from request: LanguageModelExecutorGenerationRequest,
        model: Model
    ) throws -> AFMRequest {
        try Self.request(
            from: request,
            model: model,
            metadataAccessorAvailable: requestMetadataAccessorAvailable
        )
    }

    static func request<Model: AFMFoundationModelsModelConfiguration>(
        from request: LanguageModelExecutorGenerationRequest,
        model: Model,
        metadataAccessorAvailable: Bool
    ) throws -> AFMRequest {
        var temperature = request.generationOptions.temperature
        var topP: Double?
        var topK: Int?
        var seed: Int?
        if let mode = request.generationOptions.samplingMode?.kind {
            switch mode {
            case .greedy:
                temperature = 0
            case .top(let value, let randomSeed):
                topK = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            case .nucleus(let value, let randomSeed):
                topP = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            @unknown default:
                break
            }
        }

        let toolCallingMode: String?
        switch request.generationOptions.toolCallingMode?.kind {
        case .allowed: toolCallingMode = "allowed"
        case .required: toolCallingMode = "required"
        case .disallowed: toolCallingMode = "disallowed"
        case nil: toolCallingMode = nil
        @unknown default: toolCallingMode = nil
        }

        var metadata: [String: AFMJSONValue] = [
            "includeSchemaInPrompt": .bool(
                request.contextOptions.includeSchemaInPrompt ?? true
            )
        ]
        if let toolCallingMode {
            metadata["toolCallingMode"] = .string(toolCallingMode)
        }
        let explicitReasoningRequested = request.contextOptions.reasoningLevel != nil
        if let reasoningLevel = request.contextOptions.reasoningLevel {
            switch reasoningLevel {
            case .light: metadata["reasoningLevel"] = .string("light")
            case .moderate: metadata["reasoningLevel"] = .string("moderate")
            case .deep: metadata["reasoningLevel"] = .string("deep")
            case .custom(let value): metadata["reasoningLevel"] = .string(value)
            @unknown default: break
            }
        }
        if model.supportsReasoning {
            metadata["chatTemplateKwargs"] = .object([
                "enable_thinking": .bool(explicitReasoningRequested)
            ])
        }
        mergeRequestMetadata(
            into: &metadata,
            accessorAvailable: metadataAccessorAvailable,
            read: { request.metadata }
        )

        let definitions = request.generationOptions.toolCallingMode?.kind == .disallowed
            ? []
            : request.enabledToolDefinitions
        return AFMRequest(
            messages: try messages(from: request.transcript),
            tools: try tools(from: definitions),
            options: AFMGenerationOptions(
                temperature: temperature,
                maximumResponseTokens: request.generationOptions.maximumResponseTokens
                    ?? model.defaultMaximumResponseTokens,
                topP: topP,
                topK: topK,
                seed: seed,
                responseConstraint: try responseConstraint(from: request.schema)
            ),
            metadata: metadata
        )
    }

    public static func messages(from transcript: Transcript) throws -> [AFMMessage] {
        var messages: [AFMMessage] = []
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                if let content = try content(from: instructions.segments) {
                    messages.append(AFMMessage(role: .system, content: content))
                }
            case .prompt(let prompt):
                if let content = try content(from: prompt.segments) {
                    messages.append(AFMMessage(role: .user, content: content))
                }
            case .response(let response):
                if let content = try content(from: response.segments) {
                    messages.append(AFMMessage(role: .assistant, content: content))
                }
            case .reasoning(let reasoning):
                if let content = try content(from: reasoning.segments) {
                    messages.append(AFMMessage(role: .assistant, content: content))
                }
            case .toolCalls(let toolCalls):
                messages.append(
                    AFMMessage(
                        role: .assistant,
                        content: [],
                        toolCalls: toolCalls.map {
                            AFMToolCall(
                                id: $0.id,
                                name: $0.toolName,
                                arguments: $0.arguments.jsonString
                            )
                        }
                    )
                )
            case .toolOutput(let output):
                messages.append(
                    AFMMessage(
                        role: .tool,
                        content: try content(from: output.segments) ?? [],
                        name: output.toolName,
                        toolCallID: output.id
                    )
                )
            @unknown default:
                throw unsupported(entry: entry, description: "Unknown transcript entry type.")
            }
        }
        return messages
    }

    public static func tools(
        from definitions: [Transcript.ToolDefinition]
    ) throws -> [AFMToolDefinition] {
        try definitions.map {
            AFMToolDefinition(
                name: $0.name,
                description: $0.description,
                inputSchema: try afmJSONValue(from: $0.parameters)
            )
        }
    }

    public static func responseConstraint(
        from schema: GenerationSchema?
    ) throws -> AFMResponseConstraint? {
        guard let schema else { return nil }
        return .jsonSchema(
            name: schema.name,
            schema: try afmJSONValue(from: schema),
            strict: true
        )
    }

    public static func foundationMetadata(
        _ values: [String: AFMJSONValue]
    ) -> [String: any Sendable & Codable & Equatable] {
        values.reduce(into: [:]) { result, item in
            switch item.value {
            case .null: result[item.key] = "null"
            case .bool(let value): result[item.key] = value
            case .integer(let value): result[item.key] = value
            case .number(let value): result[item.key] = value
            case .string(let value): result[item.key] = value
            case .array, .object: result[item.key] = String(describing: item.value)
            }
        }
    }

    private static func content(
        from segments: [Transcript.Segment]
    ) throws -> [AFMContentPart]? {
        let parts = try segments.flatMap { segment -> [AFMContentPart] in
            switch segment {
            case .text(let text):
                return [.text(text.content)]
            case .structure(let structure):
                return [.text(structure.content.jsonString)]
            case .attachment(let attachment):
                var result: [AFMContentPart] = []
                if let label = attachment.label, !label.isEmpty {
                    result.append(.text(label))
                }
                switch attachment.content {
                case .image(let image):
                    if let url = image.url {
                        result.append(.reference(url))
                    } else {
                        result.append(.data(mimeType: "image/png", value: try pngData(image)))
                    }
                @unknown default:
                    throw LanguageModelError.unsupportedTranscriptContent(
                        .init(
                            unsupportedContent: [],
                            debugDescription: "Unknown transcript attachment type."
                        )
                    )
                }
                return result
            case .custom(let custom):
                return [.text(customContent(custom))]
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [],
                        debugDescription: "Unknown transcript segment type."
                    )
                )
            }
        }
        return parts.isEmpty ? nil : parts
    }

    private static func customContent<C: Transcript.CustomSegment>(_ custom: C) -> String {
        if let data = try? JSONEncoder().encode(custom.content),
           let value = String(data: data, encoding: .utf8) {
            return value
        }
        return custom.description
    }

    private static func pngData(_ image: Transcript.ImageAttachment) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(unsupportedContent: [], debugDescription: "Could not create PNG output.")
            )
        }
        CGImageDestinationAddImage(destination, image.cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(unsupportedContent: [], debugDescription: "Could not encode PNG output.")
            )
        }
        return data as Data
    }

    private static func afmJSONValue<T: Encodable>(from value: T) throws -> AFMJSONValue {
        let data = try JSONEncoder().encode(value)
        return try afmJSONValue(JSONSerialization.jsonObject(with: data))
    }

    private static func afmJSONValue(_ value: Any) throws -> AFMJSONValue {
        switch value {
        case is NSNull: return .null
        case let value as Bool: return .bool(value)
        case let value as Int: return .integer(value)
        case let value as NSNumber: return .number(value.doubleValue)
        case let value as String: return .string(value)
        case let values as [Any]: return .array(try values.map(afmJSONValue))
        case let values as [String: Any]:
            return .object(try values.mapValues(afmJSONValue))
        default:
            throw AFMError.invalidRequest("Unsupported JSON schema value: \(type(of: value))")
        }
    }

    private static func afmMetadata(
        _ values: [String: any Sendable & Codable & Equatable]
    ) -> [String: AFMJSONValue] {
        var result: [String: AFMJSONValue] = [:]
        for (key, value) in values {
            switch value {
            case let value as Bool: result[key] = .bool(value)
            case let value as Int: result[key] = .integer(value)
            case let value as Double: result[key] = .number(value)
            case let value as String: result[key] = .string(value)
            default: result[key] = .string(String(describing: value))
            }
        }
        return result
    }

    static func mergeRequestMetadata(
        into metadata: inout [String: AFMJSONValue],
        accessorAvailable: Bool,
        read: () -> [String: any Sendable & Codable & Equatable]
    ) {
        guard accessorAvailable else { return }
        for (key, value) in afmMetadata(read()) {
            metadata[key] = value
        }
    }

    // Xcode 27 Beta 3 declares this getter, but some macOS 27 beta runtimes do
    // not export it. Calling the weak-linked accessor in those runtimes jumps
    // to address zero, so metadata forwarding must be capability-gated.
    static let requestMetadataAccessorAvailable = AFMRuntimeSymbolResolver.process.contains(
        "$s16FoundationModels38LanguageModelExecutorGenerationRequestV8metadataSDySSSe_SESQs8SendablepGvg"
    )

    private static func unsupported(
        entry: Transcript.Entry,
        description: String
    ) -> LanguageModelError {
        LanguageModelError.unsupportedTranscriptContent(
            .init(unsupportedContent: [entry], debugDescription: description)
        )
    }
}

#endif
