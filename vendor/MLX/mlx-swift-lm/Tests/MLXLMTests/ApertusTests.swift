import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest
import Tokenizers

final class ApertusTests: XCTestCase {
    func testQuantizedKVCacheSupportsPrefillAndDecode() {
        let model = ApertusModel(ApertusConfiguration(
            hiddenSize: 128, intermediateSize: 256, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, vocabSize: 32))
        let cache = QuantizedKVCache(groupSize: 64, bits: 4)
        let prefill = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: [cache])
        XCTAssertTrue(prefill.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(cache.offset, 3)
        let decode = model(MLXArray([4]).reshaped([1, 1]), cache: [cache])
        XCTAssertTrue(decode.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(cache.offset, 4)
    }

    func testPythonReferenceTokenParity() async throws {
        guard let fixture = ProcessInfo.processInfo.environment["APERTUS_REFERENCE_JSONL"] else {
            throw XCTSkip("Set APERTUS_REFERENCE_JSONL to a same-checkpoint Python reference run")
        }
        let records = try String(contentsOfFile: fixture, encoding: .utf8)
            .split(separator: "\n").map {
                try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
            }
        let path = try XCTUnwrap(records.first?["model"] as? String)
        let context = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: path)))
        for record in records.dropFirst() {
            let name = try XCTUnwrap(record["case"] as? String)
            let inputTokens = try XCTUnwrap(record["input_tokens"] as? [Int])
            let expected = try XCTUnwrap(record["output_tokens"] as? [Int])
                .filter { !context.resolvedEOSTokenIds.contains($0) }
            var actual = [Int]()
            _ = try generate(input: LMInput(tokens: MLXArray(inputTokens)),
                parameters: GenerateParameters(maxTokens: 256, temperature: 0), context: context
            ) { token in actual.append(token); return .more }
            print("APERTUS REFERENCE \(name): \(context.tokenizer.decode(tokens: actual))")
            XCTAssertEqual(actual, expected, "Same-checkpoint, same-token-input parity: \(name)")
        }
    }

    func testLocalCheckpointNativeToolCall() async throws {
        guard let path = ProcessInfo.processInfo.environment["APERTUS_TEST_MODEL"] else {
            throw XCTSkip("Set APERTUS_TEST_MODEL to a local checkpoint")
        }
        let context = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: path)))
        let tools: [ToolSpec] = [["type": "function", "function": [
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": ["type": "object", "properties": [
                "city": ["type": "string"]
            ], "required": ["city"]] as ToolSpec
        ] as ToolSpec]]
        let input = try await context.processor.prepare(input: UserInput(
            prompt: "Call get_weather exactly once for Tokyo. Call no other tools.",
            tools: tools, additionalContext: ["enable_thinking": true]))
        let rendered = context.tokenizer.decode(tokens: input.text.tokens.asArray(Int.self))
        XCTAssertTrue(rendered.contains("type get_weather"), rendered)
        var tokens = [Int]()
        _ = try generate(input: input,
            parameters: GenerateParameters(maxTokens: 256, temperature: 0), context: context
        ) { token in tokens.append(token); return .more }
        let output = context.tokenizer.decode(tokens: tokens)
        print("APERTUS NATIVE TOOL OUTPUT: \(output)")
        let calls = ApertusToolCallParser().parseCalls(content: output)
        XCTAssertEqual(calls.map(\.function.name), ["get_weather"], output)
        var streamedCalls = [ToolCall]()
        var visible = ""
        for await event in try generate(input: input,
            parameters: GenerateParameters(maxTokens: 256, temperature: 0), context: context) {
            switch event {
            case .toolCall(let call): streamedCalls.append(call)
            case .chunk(let text): visible += text
            default: break
            }
        }
        XCTAssertEqual(streamedCalls.map(\.function.name), ["get_weather"])
        XCTAssertFalse(visible.contains("<|tools_prefix|>"))
    }

    func testBFloat16ActivationMatchesReferenceAfterWeightLoading() throws {
        let activation = ApertusXIELU()
        let input = MLXArray([Float(-8), -2, -0.2, -0.01, -0.001, 0, 1, 2])
            .asType(.bfloat16)
        let original = activation(input)
        eval(original)
        let alphaP = MLXArray([Float(-0.75)]).asType(.bfloat16)
        let alphaN = MLXArray([Float(0.25)]).asType(.bfloat16)
        let beta = MLXArray(Float(0.5)).asType(.bfloat16)
        let eps = MLXArray(Float(-1e-6)).asType(.bfloat16)
        try activation.update(parameters: ModuleParameters.unflattened([
            "alpha_p": alphaP, "alpha_n": alphaN, "beta": beta, "eps": eps
        ]), verify: .all)
        // Reference: ml-explore/mlx-lm, mlx_lm/models/activations.py, xielu.
        let expected = MLX.where(input .> 0,
            softplus(alphaP) * square(input) + beta * input,
            (expm1(minimum(input, eps)) - input) * (beta + softplus(alphaN)) + beta * input)
        let actual = activation(input)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0)
        XCTAssertGreaterThan(abs(actual - original).max().item(Float.self), 0.1)
    }

    func testLocalCheckpointGeneration() async throws {
        guard let path = ProcessInfo.processInfo.environment["APERTUS_TEST_MODEL"] else {
            throw XCTSkip("Set APERTUS_TEST_MODEL to a local Apertus checkpoint; no automatic downloads")
        }
        let context = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: path)))
        let input = try await context.processor.prepare(input: UserInput(
            prompt: "What is the capital of Switzerland? Answer in one sentence."))
        var tokens = [Int]()
        let info = try generate(
            input: input, parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: context
        ) { token in
            tokens.append(token)
            return .more
        }
        let output = context.tokenizer.decode(tokens: tokens)
        print("APERTUS CHECKPOINT: \(path)\nOUTPUT: \(output)\nTIMING: \(info)")
        XCTAssertTrue(output.lowercased().contains("bern"), output)
    }

    func testIntegerAndFloatRopeMetadataProduceIdenticalLogits() throws {
        var config = ApertusConfiguration(
            hiddenSize: 16, intermediateSize: 32, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, vocabSize: 32,
            ropeTheta: 12_000_000,
            ropeScaling: ["type": .string("llama3"), "factor": .float(8),
                "low_freq_factor": .float(1), "high_freq_factor": .float(4),
                "original_max_position_embeddings": .float(8192)])
        let floatModel = ApertusModel(config)
        config.ropeScaling = ["rope_type": .string("llama3"), "factor": .int(8),
            "low_freq_factor": .int(1), "high_freq_factor": .int(4),
            "original_max_position_embeddings": .int(8192)]
        // Exercise the actual JSON checkpoint decoding path as well as the alias.
        let decoded = try JSONDecoder().decode(
            ApertusConfiguration.self, from: JSONEncoder().encode(config))
        let integerModel = ApertusModel(decoded)
        // Derived RoPE frequencies are configuration state, not checkpoint weights.
        let weights = floatModel.parameters().flattened().filter { !$0.0.contains(".rope.") }
        try integerModel.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        let input = MLXArray(Array(0..<32)).reshaped([1, 32])
        let expected = floatModel(input)
        let actual = integerModel(input)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-6)
    }

    func testActivationParametersUseCheckpointNames() throws {
        let model = ApertusModel(ApertusConfiguration(
            hiddenSize: 16, intermediateSize: 32, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, vocabSize: 32))
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for name in ["alpha_p", "alpha_n", "beta", "eps"] {
            XCTAssertNotNil(parameters["model.layers.0.mlp.act_fn.\(name)"], name)
        }
    }
}
