import AFMKitCore
import AFMKitMLX
import Foundation

@main
struct AFMKitQuickstart {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count >= 2 else {
            print("Usage: afmkit-quickstart <Hugging-Face-model> <prompt>")
            print("Example: afmkit-quickstart mlx-community/Qwen3.8-27B-4bit \"Why is the sky blue?\"")
            return
        }

        let modelID = AFMModelID(rawValue: arguments[0])
        let prompt = arguments.dropFirst().joined(separator: " ")
        let registry = AFMProviderRegistry()
        try registry.register(AFMMLXProviderFactory())

        let model = try registry.makeModel(
            providerID: AFMMLXProviderFactory.providerID,
            modelID: modelID,
            configuration: AFMProviderConfiguration(values: [
                "enablePrefixCaching": .bool(true),
                "maxConcurrent": .integer(1)
            ])
        )

        let descriptor = try await model.load { progress in
            let percentage = Int((progress * 100).rounded())
            FileHandle.standardError.write(Data("\rLoading \(percentage)%".utf8))
        }
        FileHandle.standardError.write(Data("\rLoaded \(descriptor.displayName)\n".utf8))

        let request = AFMRequest(
            messages: [AFMMessage(role: .user, text: prompt)],
            options: AFMGenerationOptions(maximumResponseTokens: 512)
        )

        for try await event in model.streamResponse(to: request) {
            render(event)
        }
        await model.unload()
    }

    private static func render(_ event: AFMGenerationEvent) {
        switch event {
        case .responseText(let action, let text, _):
            if action == .replace {
                print("\n[response replaced]\n", terminator: "")
            }
            print(text, terminator: "")
            fflush(stdout)
        case .reasoningText(let action, let text, _):
            if action == .replace {
                FileHandle.standardError.write(Data("\n[reasoning replaced]\n".utf8))
            }
            FileHandle.standardError.write(Data(text.utf8))
        case .toolCall(let call, let stage):
            FileHandle.standardError.write(
                Data("\n[tool \(call.name): \(String(describing: stage))]\n".utf8)
            )
        case .usage(let usage):
            FileHandle.standardError.write(
                Data("\n[usage input=\(usage.inputTokens) output=\(usage.outputTokens)]\n".utf8)
            )
        case .completed(let reason):
            print("\n[completed: \(reason.rawValue)]")
        case .tokenLogprobs, .metadata, .custom:
            break
        }
    }
}
