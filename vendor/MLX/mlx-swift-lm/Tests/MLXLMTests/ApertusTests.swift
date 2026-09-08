import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class ApertusTests: XCTestCase {
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
        let integerModel = ApertusModel(config)
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
