#if canImport(FoundationModels)
import AFMKitCore
import Foundation
import FoundationModels
import ImageIO

@available(macOS 27.0, *)
/// Configuration keys understood by `AFMFoundationProviderFactory` models.
public enum AFMFoundationProviderConfigurationKeys {
    /// Additional instructions prepended to system messages in each request.
    public static let systemPrompt = "systemPrompt"
    /// PCC reasoning level: automatic, light, moderate, or deep.
    public static let reasoningLevel = "reasoningLevel"
}

@available(macOS 27.0, *)
enum AFMFoundationRequestAttachment: Equatable, Sendable {
    case data(mimeType: String, value: Data, label: String)
    case reference(URL, label: String)
}

@available(macOS 27.0, *)
struct AFMFoundationRequestPlan: Equatable, Sendable {
    let instructions: String
    let conversation: String
    let attachments: [AFMFoundationRequestAttachment]
    let generationOptions: AFMFoundationGenerationOptionPlan
    let reasoningLevel: AFMFoundationReasoningLevel?
    let requestedToolNames: [String]
}

@available(macOS 27.0, *)
enum AFMFoundationProviderRequestAdapter {
    static func plan(
        request: AFMRequest,
        provider: AFMFoundationNativeProviderKind,
        configuredSystemPrompt: String,
        configuredReasoningLevel: AFMFoundationReasoningLevel,
        availableToolNames: Set<String>
    ) throws -> AFMFoundationRequestPlan {
        guard !request.messages.isEmpty else {
            throw AFMError.invalidRequest("At least one message is required.")
        }

        try validateSupportedOptions(request.options)

        var systemParts: [String] = []
        if !configuredSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemParts.append(configuredSystemPrompt)
        }

        var lines: [String] = []
        var attachments: [AFMFoundationRequestAttachment] = []
        var attachmentIndex = 0

        for message in request.messages {
            if message.role == .system {
                let text = try textOnlyContent(message.content, context: "system messages")
                if !text.isEmpty {
                    systemParts.append(text)
                }
                continue
            }

            let role = roleLabel(for: message)
            var renderedParts: [String] = []
            for part in message.content {
                switch part {
                case .text(let text):
                    renderedParts.append(text)
                case .data(let mimeType, let value):
                    guard mimeType.lowercased().hasPrefix("image/") else {
                        throw AFMError.unsupportedCapability(
                            "Foundation Models attachments currently support image data only."
                        )
                    }
                    attachmentIndex += 1
                    let label = "image-\(attachmentIndex)"
                    attachments.append(.data(mimeType: mimeType, value: value, label: label))
                    renderedParts.append("[Attached image: \(label)]")
                case .reference(let url):
                    attachmentIndex += 1
                    let label = "image-\(attachmentIndex)"
                    attachments.append(.reference(url, label: label))
                    renderedParts.append("[Attached image: \(label)]")
                case .custom(let type, _):
                    throw AFMError.unsupportedCapability(
                        "Foundation Models does not support custom content part '\(type)'."
                    )
                }
            }

            if !message.toolCalls.isEmpty {
                let calls = message.toolCalls.map {
                    "\($0.name)(\($0.arguments)) [id=\($0.id)]"
                }
                renderedParts.append("Tool calls: " + calls.joined(separator: ", "))
            }
            lines.append("\(role): \(renderedParts.joined(separator: "\n"))")
        }
        lines.append("Assistant:")

        let requestedToolNames = request.tools.map(\.name)
        let unknownTools = Set(requestedToolNames).subtracting(availableToolNames).sorted()
        guard unknownTools.isEmpty else {
            throw AFMError.invalidRequest(
                "No executable Foundation Models tool was supplied for: \(unknownTools.joined(separator: ", "))."
            )
        }

        let toolChoice = request.metadata["toolChoice"]?.stringValue ?? "auto"
        guard ["auto", "none", "required"].contains(toolChoice) else {
            throw AFMError.invalidRequest("toolChoice must be auto, none, or required.")
        }
        let toolsEnabled = !requestedToolNames.isEmpty && toolChoice != "none"
        let requiresTools = toolsEnabled && toolChoice == "required"

        let usesProviderDefaults = request.options.temperature == nil
            && request.options.topP == nil
            && request.options.maximumResponseTokens == nil
        let parameterPlan = AFMFoundationGenerationParameters(
            useProviderDefaults: usesProviderDefaults,
            temperature: request.options.temperature ?? 0.7,
            topP: request.options.topP ?? 0.9,
            maxTokens: request.options.maximumResponseTokens
        )
        let generationOptions = AFMFoundationGenerationOptionsPolicy.plan(
            from: parameterPlan,
            allowsToolCalling: !availableToolNames.isEmpty,
            toolsEnabled: toolsEnabled,
            requiresToolCalling: requiresTools
        )

        let reasoningLevel = try requestedReasoningLevel(
            request: request,
            configured: configuredReasoningLevel
        )
        if provider == .appleOnDevice,
           reasoningLevel != nil,
           reasoningLevel != .automatic {
            throw AFMError.unsupportedCapability(
                "Reasoning levels are supported by Private Cloud Compute, not the on-device model."
            )
        }

        return AFMFoundationRequestPlan(
            instructions: systemParts.joined(separator: "\n\n"),
            conversation: lines.joined(separator: "\n\n"),
            attachments: attachments,
            generationOptions: generationOptions,
            reasoningLevel: provider == .privateCloudCompute ? reasoningLevel : nil,
            requestedToolNames: toolsEnabled ? requestedToolNames : []
        )
    }

    static func reasoningLevel(from value: AFMJSONValue?) throws -> AFMFoundationReasoningLevel {
        guard let value else { return .automatic }
        guard case .string(let rawValue) = value else {
            throw AFMError.invalidRequest("reasoningLevel must be a string.")
        }
        switch rawValue.lowercased() {
        case "automatic", "auto": return .automatic
        case "light", "low": return .light
        case "moderate", "medium": return .moderate
        case "deep", "high": return .deep
        default:
            throw AFMError.invalidRequest(
                "reasoningLevel must be automatic, light, moderate, or deep."
            )
        }
    }

    private static func requestedReasoningLevel(
        request: AFMRequest,
        configured: AFMFoundationReasoningLevel
    ) throws -> AFMFoundationReasoningLevel? {
        if let requestValue = request.metadata[AFMFoundationProviderConfigurationKeys.reasoningLevel] {
            return try reasoningLevel(from: requestValue)
        }
        return configured
    }

    private static func validateSupportedOptions(_ options: AFMGenerationOptions) throws {
        var unsupported: [String] = []
        if options.topK != nil { unsupported.append("topK") }
        if options.minP != nil { unsupported.append("minP") }
        if options.repetitionPenalty != nil { unsupported.append("repetitionPenalty") }
        if options.presencePenalty != nil { unsupported.append("presencePenalty") }
        if options.seed != nil { unsupported.append("seed") }
        if options.logprobs != nil { unsupported.append("logprobs") }
        if options.topLogprobs != nil { unsupported.append("topLogprobs") }
        if options.reasoningEnabled != nil { unsupported.append("reasoningEnabled") }
        if !options.stopSequences.isEmpty { unsupported.append("stopSequences") }
        if case .grammar? = options.responseConstraint { unsupported.append("grammar") }
        if case .jsonObject? = options.responseConstraint { unsupported.append("jsonObject") }
        guard unsupported.isEmpty else {
            throw AFMError.unsupportedCapability(
                "Foundation Models does not support these AFM options: \(unsupported.joined(separator: ", "))."
            )
        }
    }

    private static func textOnlyContent(
        _ content: [AFMContentPart],
        context: String
    ) throws -> String {
        var parts: [String] = []
        for part in content {
            guard case .text(let value) = part else {
                throw AFMError.invalidRequest("Only text is allowed in \(context).")
            }
            parts.append(value)
        }
        return parts.joined(separator: "\n")
    }

    private static func roleLabel(for message: AFMMessage) -> String {
        switch message.role {
        case .system: return "System"
        case .user: return message.name.map { "User (\($0))" } ?? "User"
        case .assistant: return message.name.map { "Assistant (\($0))" } ?? "Assistant"
        case .tool:
            let name = message.name ?? "tool"
            return message.toolCallID.map { "Tool \(name) [id=\($0)]" } ?? "Tool \(name)"
        }
    }
}

@available(macOS 27.0, *)
private extension AFMJSONValue {
    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }
}

@available(macOS 27.0, *)
struct AFMFoundationPromptImage: PromptRepresentable, @unchecked Sendable {
    enum Source: @unchecked Sendable {
        case image(CGImage)
        case url(URL)
    }

    let source: Source
    let label: String

    var promptRepresentation: Prompt {
        Prompt {
            "The attached image is labeled '\(label)'."
            switch source {
            case .image(let image):
                Attachment(image).label(label)
            case .url(let url):
                Attachment(imageURL: url).label(label)
            }
        }
    }
}

@available(macOS 27.0, *)
extension AFMFoundationRequestPlan {
    func prompt() throws -> Prompt {
        let images = try attachments.map { attachment -> AFMFoundationPromptImage in
            switch attachment {
            case .reference(let url, let label):
                return AFMFoundationPromptImage(source: .url(url), label: label)
            case .data(_, let value, let label):
                guard let source = CGImageSourceCreateWithData(value as CFData, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw AFMError.invalidRequest("Attached image '\(label)' could not be decoded.")
                }
                return AFMFoundationPromptImage(source: .image(image), label: label)
            }
        }
        return Prompt {
            conversation
            for image in images {
                image
            }
        }
    }
}
#endif
