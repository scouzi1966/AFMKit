// Copyright © 2025 Apple Inc.

import Foundation
@testable import MLXLMCommon
import XCTest

public class BaseConfigurationTests: XCTestCase {

    func testQuantization() throws {
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 128,
                    "bits": 4
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.quantization, .init(groupSize: 128, bits: 4))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"), .init(groupSize: 128, bits: 4))
    }

    func testHeterogenousQuantization() throws {
        // from https://huggingface.co/mlx-community/Qwen3-1.7B-4bit-AWQ/blob/main/config.json#L20
        let json =
            """
            {
                "model_type": "Test",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.embed_tokens": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.0.self_attn.q_norm": false,
                    "true_layer": true
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.quantization, .init(groupSize: 64, bits: 4))

        // a random layer -- no specific configuration gets default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "x"),
            .init(groupSize: 64, bits: 4))

        // layer with an override
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.embed_tokens"),
            .init(groupSize: 32, bits: 4))

        // layer with an override -- not quant
        XCTAssertNil(
            config.perLayerQuantization?.quantization(layer: "model.layers.0.self_attn.q_norm"))

        // layer with an override -- true, use the default
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "true_layer"),
            .init(groupSize: 64, bits: 4))
    }

    func testQwenNextPLEShardQuantizationFollowsSanitizedModulePath() throws {
        let json =
            """
            {
                "model_type": "qwen4_exp",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shard_7": {
                        "group_size": 32,
                        "bits": 4
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(
                layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.7"),
            .init(groupSize: 32, bits: 4))
    }

    func testQwenNextPLEShardQuantizationAliasIsBidirectional() throws {
        let json =
            """
            {
                "model_type": "qwen4_exp",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shards.11": {
                        "group_size": 32,
                        "bits": 4
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(
                layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shard_11"),
            .init(groupSize: 32, bits: 4))
    }

    func testQwenNextPLEShardQuantizationAliasPreservesSkipAndExactPrecedence() throws {
        let json =
            """
            {
                "model_type": "qwen4_exp",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shard_3": false,
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shard_5": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shards.5": {
                        "group_size": 16,
                        "bits": 4
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(json.utf8))

        XCTAssertNil(
            config.perLayerQuantization?.quantization(
                layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.3"))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(
                layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.5"),
            .init(groupSize: 16, bits: 4))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(
                layer: "model.layers.1.ple.ple_embedding.ngram_embedding.shard_5"),
            .init(groupSize: 32, bits: 4))
    }

    func testQwenNextPLEShardQuantizationAliasPreservesNamespacesAndUnrelatedPaths() throws {
        let json =
            """
            {
                "model_type": "qwen4_exp",
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "language_model.model.layers.2.ple.ple_embedding.ngram_embedding.shard_9": {
                        "group_size": 32,
                        "bits": 4
                    },
                    "model.layers.1.shard_7": {
                        "group_size": 32,
                        "bits": 4
                    }
                }
            }
            """

        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(
            config.perLayerQuantization?.quantization(
                layer: "layers.2.ple.ple_embedding.ngram_embedding.shards.9"),
            .init(groupSize: 32, bits: 4))
        XCTAssertEqual(
            config.perLayerQuantization?.quantization(layer: "model.layers.1.shards.7"),
            .init(groupSize: 64, bits: 4))
    }

    func testCheckpointQuantizationInfersHeterogeneousAffineBitWidth() {
        let defaultQuantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)

        XCTAssertEqual(
            resolveCheckpointQuantization(
                layer: "lm_head",
                weightShape: [248_320, 640],
                scaleShape: [248_320, 40],
                quantization: defaultQuantization,
                perLayerQuantization: nil),
            .init(groupSize: 64, bits: 8, mode: .affine))
    }

    func testCheckpointQuantizationKeepsExplicitOverrideAuthoritative() {
        let explicit = BaseConfiguration.Quantization(groupSize: 32, bits: 4)
        let perLayer = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: ["lm_head": .quantize(explicit)])

        XCTAssertEqual(
            resolveCheckpointQuantization(
                layer: "lm_head",
                weightShape: [248_320, 640],
                scaleShape: [248_320, 40],
                quantization: nil,
                perLayerQuantization: perLayer),
            explicit)
    }

    func testCheckpointQuantizationKeepsExplicitSkipAuthoritative() {
        let perLayer = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: ["lm_head": .skip])

        XCTAssertNil(
            resolveCheckpointQuantization(
                layer: "lm_head",
                weightShape: [248_320, 640],
                scaleShape: [248_320, 40],
                quantization: nil,
                perLayerQuantization: perLayer))
    }

    func testAffineQuantizationInferenceRejectsIncompatibleGeometry() {
        XCTAssertNil(
            inferAffineQuantization(
                weightShape: [8, 640], scaleShape: [7, 40], groupSize: 64))
        XCTAssertNil(
            inferAffineQuantization(
                weightShape: [8, 560], scaleShape: [8, 40], groupSize: 64))
    }

}
