import AFMKitApple
import AFMKitCore
import AFMKitInference
import AFMKitMLX
import AFMOpenAICompat
import Darwin
import Foundation

@main
struct AFMKitQuickstart {
    static func main() async {
        do {
            try await run()
        } catch {
            FileHandle.standardError.write(Data("AFMKitQuickstart error: \(error)\n".utf8))
            exit(EXIT_FAILURE)
        }
    }

    private static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let provider = arguments.first else {
            printUsage()
            return
        }

        switch provider {
        case "mlx":
            guard arguments.count >= 3 else {
                printUsage()
                return
            }
            try await runMLX(modelID: arguments[1], prompt: arguments.dropFirst(2).joined(separator: " "))
        case "apple-on-device", "apple-pcc":
            guard arguments.count >= 2 else {
                printUsage()
                return
            }
            guard #available(macOS 27.0, *) else {
                throw AFMError.unavailable("Apple Foundation Models require macOS 27 or newer.")
            }
            try await runApple(
                privateCloudCompute: provider == "apple-pcc",
                prompt: arguments.dropFirst().joined(separator: " ")
            )
        default:
            printUsage()
        }
    }

    private static func runMLX(modelID: String, prompt: String) async throws {
        let registry = AFMProviderRegistry()
        try registry.register(AFMMLXProviderFactory())
        let model = try registry.makeModel(
            providerID: AFMMLXProviderFactory.providerID,
            modelID: AFMModelID(rawValue: modelID),
            configuration: AFMProviderConfiguration(values: [
                "enablePrefixCaching": .bool(true),
                "maxConcurrent": .integer(1),
            ])
        )
        try await generate(model: model, prompt: prompt, reasoningEnabled: false)
    }

    @available(macOS 27.0, *)
    private static func runApple(
        privateCloudCompute: Bool,
        prompt: String
    ) async throws {
        let registry = AFMProviderRegistry()
        try registry.register(AFMFoundationProviderFactory())
        let model = try registry.makeModel(
            providerID: AFMFoundationProviderFactory.providerID,
            modelID: privateCloudCompute
                ? AFMFoundationProviderFactory.privateCloudComputeModelID
                : AFMFoundationProviderFactory.onDeviceModelID,
            configuration: .init()
        )
        try await generate(model: model, prompt: prompt)
    }

    private static func generate(
        model: AnyAFMModel,
        prompt: String,
        reasoningEnabled: Bool? = nil
    ) async throws {
        let engine = try AFMEngine(model: model)
        let descriptor = try await engine.load { progress in
            let percentage = Int((progress * 100).rounded())
            FileHandle.standardError.write(Data("\rLoading \(percentage)%".utf8))
        }
        FileHandle.standardError.write(Data("\rLoaded \(descriptor.displayName)\n".utf8))

        for try await event in engine.streamEvents(
            to: [Message(role: "user", content: prompt)],
            GenerationConfig(maxTokens: 512, reasoningEnabled: reasoningEnabled)
        ) {
            render(event)
        }
        await engine.unload()
    }

    private static func printUsage() {
        print("Usage:")
        print("  afmkit-quickstart mlx <Hugging-Face-model> <prompt>")
        print("  afmkit-quickstart apple-on-device <prompt>")
        print("  afmkit-quickstart apple-pcc <prompt>")
    }

    private static func render(_ event: AFMStreamEvent) {
        switch event {
        case .text(let action, let text, _):
            if action == .replace { print("\n[response replaced]\n", terminator: "") }
            print(text, terminator: "")
            fflush(stdout)
        case .reasoning(let action, let text, _):
            if action == .replace {
                FileHandle.standardError.write(Data("\n[reasoning replaced]\n".utf8))
            }
            FileHandle.standardError.write(Data(text.utf8))
        case .toolCall(let call, let stage):
            FileHandle.standardError.write(
                Data("\n[tool \(call.name): \(String(describing: stage))]\n".utf8)
            )
        case .usage(let input, let output, _, let reasoning):
            FileHandle.standardError.write(
                Data("\n[usage input=\(input) output=\(output) reasoning=\(reasoning)]\n".utf8)
            )
        case .completed(let reason):
            print("\n[completed: \(reason.rawValue)]")
        case .tokenLogprobs, .metadata, .custom:
            break
        }
    }
}
