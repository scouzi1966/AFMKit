import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class ApertusTests: XCTestCase {
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
