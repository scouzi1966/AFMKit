import Foundation
import MLX
@testable import AFMKitMLX
import XCTest

final class GLM5NextCheckpointConverterTests: XCTestCase {
    private struct FixtureTensor {
        let dtype: String
        let shape: [Int]
        let data: Data
    }

    func testMLXSwiftDecodesKnownE4M3Bytes() {
        let raw = MLXArray([UInt8(0x00), 0x38, 0x40, 0xb8, 0x7e])
        let decoded = MLX.fromFP8(raw, dtype: .float32)
        MLX.eval(decoded)

        XCTAssertEqual(decoded.dtype, .float32)
        XCTAssertEqual(decoded.asArray(Float.self), [0, 1, 2, -1, 448])
    }

    func testBlockInverseScalesAreAppliedInFloat32() throws {
        let raw = MLXArray([UInt8](repeating: 0x38, count: 256 * 256))
            .reshaped(256, 256)
        let scales = MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)
        let decoded = try GLM5NextCheckpointConverter.dequantizeFP8(
            raw: raw,
            scaleInverse: scales,
            weightShape: [256, 256],
            scaleShape: [2, 2])
        MLX.eval(decoded)

        XCTAssertEqual(decoded.dtype, .float32)
        XCTAssertEqual(decoded[0, 0].item(Float.self), 1)
        XCTAssertEqual(decoded[0, 255].item(Float.self), 2)
        XCTAssertEqual(decoded[255, 0].item(Float.self), 3)
        XCTAssertEqual(decoded[255, 255].item(Float.self), 4)
    }

    func testBlockScaleShapeMismatchFailsClosed() {
        XCTAssertThrowsError(try GLM5NextCheckpointConverter.dequantizeFP8(
            raw: MLXArray([UInt8](repeating: 0x38, count: 128 * 128)).reshaped(128, 128),
            scaleInverse: MLXArray([Float(1), 2]).reshaped(1, 2),
            weightShape: [128, 128],
            scaleShape: [1, 2])) { error in
                XCTAssertTrue(error.localizedDescription.contains("does not match"))
            }
    }

    func testNonFiniteFP8ScaleIsRejected() {
        XCTAssertThrowsError(try GLM5NextCheckpointConverter.dequantizeFP8(
            raw: MLXArray([UInt8](repeating: 0x38, count: 128 * 128)).reshaped(128, 128),
            scaleInverse: MLXArray(Float.nan).reshaped(1, 1),
            weightShape: [128, 128],
            scaleShape: [1, 1])) { error in
                XCTAssertTrue(error.localizedDescription.contains("non-finite"))
            }
    }

    func testAffineFourBitRoundTripHasBoundedError() throws {
        let source = MLXArray((0 ..< 128 * 128).map {
            Float(($0 % 251) - 125) / 32
        }).reshaped(128, 128)
        let quantized = try GLM5NextCheckpointConverter.affineQuantize(source)
        let restored = dequantized(
            quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            groupSize: 64,
            bits: 4,
            dtype: .float32)
        let maximumError = abs(restored - source).max()
        MLX.eval(maximumError)

        XCTAssertLessThan(maximumError.item(Float.self), 0.5)
        XCTAssertEqual(quantized.weight.dtype, .uint32)
        XCTAssertEqual(quantized.scales.dtype, .bfloat16)
        XCTAssertEqual(quantized.biases.dtype, .bfloat16)
    }

    func testPublishedNamespaceMapsTextVisionAndControls() {
        XCTAssertEqual(
            GLM5NextCheckpointConverter.mappedName(
                "model.language_model.layers.3.hc_attn_fn"),
            "language_model.model.layers.3.attn_hc.fn")
        XCTAssertEqual(
            GLM5NextCheckpointConverter.mappedName(
                "model.language_model.layers.0.self_attn.A_log"),
            "language_model.model.layers.0.self_attn.forget_gate.A_log")
        XCTAssertEqual(
            GLM5NextCheckpointConverter.mappedName("lm_head.weight"),
            "language_model.lm_head.weight")
        XCTAssertEqual(
            GLM5NextCheckpointConverter.mappedName(
                "model.visual.blocks.0.attn.qkv.weight"),
            "vision_model.blocks.0.attn.qkv.weight")
    }

    func testOfficialVisionConvolutionLayoutsTransposeToMLX() throws {
        XCTAssertEqual(
            try GLM5NextCheckpointConverter.visionWeightPermutation(
                sourceName: "model.visual.patch_embed.proj.weight",
                sourceShape: [1024, 3, 2, 14, 14]),
            [0, 2, 3, 4, 1])
        XCTAssertEqual(
            try GLM5NextCheckpointConverter.visionWeightPermutation(
                sourceName: "model.visual.downsample.weight",
                sourceShape: [4096, 1024, 2, 2]),
            [0, 2, 3, 1])

        let source = MLXArray((0..<(1024 * 3 * 2 * 14 * 14)).map(Float.init))
            .reshaped(1024, 3, 2, 14, 14)
        let converted = try GLM5NextCheckpointConverter.convertedVisionWeight(
            source,
            sourceName: "model.visual.patch_embed.proj.weight",
            sourceShape: source.shape)
        MLX.eval(converted)

        XCTAssertEqual(converted.shape, [1024, 2, 14, 14, 3])
        XCTAssertEqual(
            converted[7, 1, 13, 12, 2].item(Float.self),
            source[7, 2, 1, 13, 12].item(Float.self))
    }

    func testUnexpectedOfficialVisionConvolutionShapeFailsClosed() {
        XCTAssertThrowsError(try GLM5NextCheckpointConverter.visionWeightPermutation(
            sourceName: "model.visual.downsample.weight",
            sourceShape: [4, 4])) { error in
                XCTAssertTrue(error.localizedDescription.contains("downsample shape"))
            }
    }

    func testAllPublishedVisionTensorNamesMapWithoutLoss() {
        let blockLeaves = [
            "attn.k_norm.weight", "attn.proj.bias", "attn.proj.weight",
            "attn.q_norm.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.down_proj.bias", "mlp.down_proj.weight", "mlp.gate_proj.bias",
            "mlp.gate_proj.weight", "mlp.up_proj.bias", "mlp.up_proj.weight",
            "norm1.weight", "norm2.weight",
        ]
        var published = (0..<24).flatMap { block in
            blockLeaves.map { "model.visual.blocks.\(block).\($0)" }
        }
        published += [
            "model.visual.downsample.bias", "model.visual.downsample.weight",
            "model.visual.merger.down_proj.weight", "model.visual.merger.gate_proj.weight",
            "model.visual.merger.post_projection_norm.bias",
            "model.visual.merger.post_projection_norm.weight",
            "model.visual.merger.proj.weight", "model.visual.merger.up_proj.weight",
            "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
            "model.visual.post_layernorm.weight",
        ]
        let mapped = published.map(GLM5NextCheckpointConverter.mappedName)

        XCTAssertEqual(published.count, 347)
        XCTAssertEqual(Set(mapped).count, 347)
        XCTAssertTrue(mapped.allSatisfy { $0.hasPrefix("vision_model.") })
        XCTAssertTrue(mapped.contains("vision_model.merger.proj.weight"))
    }

    func testRawU8WeightIsNotInferredAsFP8() throws {
        let root = try makeFixture(rawExpertDType: "U8", includeExpertScales: false)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: root,
            sourceRevision: String(repeating: "a", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("never infers FP8 from UInt8"))
            }
    }

    func testMissingFP8ScaleIsRejected() throws {
        let root = try makeFixture(includeExpertScales: false)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: root,
            sourceRevision: String(repeating: "a", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("requires a paired F32"))
            }
    }

    func testNonF32FP8ScaleIsRejected() throws {
        let root = try makeFixture(expertScaleDType: "BF16")
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: root,
            sourceRevision: String(repeating: "a", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("requires a paired F32"))
            }
    }

    func testSafetensorHeaderLengthCannotExceedFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("invalid.safetensors")
        var declared = UInt64(513 * 1024 * 1024).littleEndian
        try withUnsafeBytes(of: &declared) { Data($0) }.write(to: url)

        XCTAssertThrowsError(try AFMSafetensorHeader(url: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Invalid SafeTensor header"))
        }
    }

    func testTinyMultimodalConversionReconstructsCrossShardExpertsAndOmitsMTP() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        let revision = String(repeating: "a", count: 40)
        let inspection = try GLM5NextCheckpointConverter.inspect(
            source: source, sourceRevision: revision)

        XCTAssertEqual(inspection.modelType, "glm5_next")
        XCTAssertEqual(inspection.shardCount, 2)
        XCTAssertGreaterThan(inspection.fp8TensorCount, 0)
        XCTAssertGreaterThan(inspection.visionTensorCount, 0)
        XCTAssertGreaterThan(inspection.omittedMTPTensorCount, 0)
        XCTAssertEqual(
            inspection.requiredDestinationFreeBytes,
            GLM5NextCheckpointConverter.minimumDestinationFreeBytes)

        try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: revision).run()

        let index = try json(at: output.appendingPathComponent(
            "model.safetensors.index.json"))
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            XCTAssertNotNil(weightMap[
                "language_model.model.layers.1.mlp.switch_mlp.\(projection).weight"])
        }
        XCTAssertNotNil(weightMap["vision_model.patch_embed.proj.weight"])
        let visionFile = try XCTUnwrap(
            weightMap["vision_model.patch_embed.proj.weight"])
        let visionArrays = try loadArrays(url: output.appendingPathComponent(visionFile))
        XCTAssertEqual(
            try XCTUnwrap(visionArrays["vision_model.patch_embed.proj.weight"]).shape,
            [1024, 2, 14, 14, 3])
        XCTAssertNotNil(weightMap[
            "language_model.model.layers.0.self_attn.conv1d.weight"])
        XCTAssertNotNil(weightMap[
            "language_model.model.layers.1.self_attn.embed_q.weight"])
        XCTAssertNotNil(weightMap[
            "language_model.model.layers.1.self_attn.unembed_out.weight"])
        XCTAssertFalse(weightMap.keys.contains { $0.contains("layers.2") })
        XCTAssertFalse(weightMap.keys.contains { $0.hasSuffix("weight_scale_inv") })

        let expertKey = "language_model.model.layers.1.mlp.switch_mlp.gate_proj"
        let expertShard = try XCTUnwrap(weightMap["\(expertKey).weight"])
        let expertArrays = try loadArrays(url: output.appendingPathComponent(expertShard))
        let restored = dequantized(
            try XCTUnwrap(expertArrays["\(expertKey).weight"]),
            scales: try XCTUnwrap(expertArrays["\(expertKey).scales"]),
            biases: try XCTUnwrap(expertArrays["\(expertKey).biases"]),
            groupSize: 64,
            bits: 4,
            dtype: .float32)
        let means = restored.mean(axes: [1, 2])
        MLX.eval(means)
        XCTAssertEqual(means.shape, [12])
        XCTAssertEqual(means[0].item(Float.self), 1, accuracy: 0.01)
        XCTAssertEqual(means[2].item(Float.self), 2, accuracy: 0.01)
        XCTAssertEqual(means[10].item(Float.self), 4, accuracy: 0.01)

        let config = try json(at: output.appendingPathComponent("config.json"))
        XCTAssertNil(config["quantization_config"])
        let provenance = try XCTUnwrap(config["afm_conversion"] as? [String: Any])
        XCTAssertEqual(provenance["source_revision"] as? String, revision)
        XCTAssertEqual(provenance["mtp_omitted"] as? Bool, true)
        XCTAssertEqual(provenance["vision_preserved"] as? Bool, true)
        let text = try XCTUnwrap(config["text_config"] as? [String: Any])
        XCTAssertEqual(text["num_nextn_predict_layers"] as? Int, 0)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("processor_config.json").path))
    }

    func testResumeRejectsChangedSourceRevision() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: String(repeating: "a", count: 40)).run()

        XCTAssertThrowsError(try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: String(repeating: "b", count: 40)).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains("Source revision changed"))
            }
    }

    func testResumeRewritesSameSizeCorruptedOutputUnit() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        let revision = String(repeating: "a", count: 40)
        let converter = GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: revision)
        try converter.run()

        let index = try json(at: output.appendingPathComponent(
            "model.safetensors.index.json"))
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        let fileName = try XCTUnwrap(weightMap["vision_model.patch_embed.proj.weight"])
        let unitURL = output.appendingPathComponent(fileName)
        let original = try Data(contentsOf: unitURL)
        var corrupted = original
        corrupted[corrupted.index(before: corrupted.endIndex)] ^= 0xff
        try corrupted.write(to: unitURL)
        XCTAssertEqual(corrupted.count, original.count)
        XCTAssertNotEqual(corrupted, original)

        try converter.run()

        XCTAssertEqual(try Data(contentsOf: unitURL), original)
    }

    func testResumeRejectsChangedConfigOrIndex() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        let revision = String(repeating: "a", count: 40)
        try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: revision).run()

        let configURL = source.appendingPathComponent("config.json")
        let originalConfig = try Data(contentsOf: configURL)
        var changedConfig = try json(at: configURL)
        changedConfig["fixture_mutation"] = true
        try JSONSerialization.data(withJSONObject: changedConfig, options: [.sortedKeys])
            .write(to: configURL)
        assertSourceMismatch(source: source, output: output, revision: revision)
        try originalConfig.write(to: configURL)

        let indexURL = source.appendingPathComponent("model.safetensors.index.json")
        var changedIndex = try json(at: indexURL)
        changedIndex["fixture_mutation"] = true
        try JSONSerialization.data(withJSONObject: changedIndex, options: [.sortedKeys])
            .write(to: indexURL)
        assertSourceMismatch(source: source, output: output, revision: revision)
    }

    func testOutputAncestorOfSourceIsRejectedBeforeOverwrite() throws {
        let container = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let fixture = try makeFixture()
        let source = container.appendingPathComponent("models/source", isDirectory: true)
        try FileManager.default.createDirectory(
            at: source.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: fixture, to: source)
        defer { try? FileManager.default.removeItem(at: container) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter(
            source: source,
            output: container,
            overwrite: true,
            sourceRevision: String(repeating: "a", count: 40)).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: source.appendingPathComponent("config.json").path))
    }

    func testSymlinkedOutputAncestorOfSourceIsRejectedBeforeOverwrite() throws {
        let container = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let fixture = try makeFixture()
        let real = container.appendingPathComponent("real", isDirectory: true)
        let source = real.appendingPathComponent("source", isDirectory: true)
        let alias = container.appendingPathComponent("alias", isDirectory: true)
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: fixture, to: source)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)
        defer { try? FileManager.default.removeItem(at: container) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter(
            source: source,
            output: alias,
            overwrite: true,
            sourceRevision: String(repeating: "a", count: 40)).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: source.appendingPathComponent("config.json").path))
    }

    func testFilesystemRootAndRootSymlinkAreRejectedWithoutDeletion() throws {
        let source = try makeFixture()
        let alias = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: alias, withDestinationURL: URL(fileURLWithPath: "/", isDirectory: true))
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: alias)
        }

        for output in [URL(fileURLWithPath: "/", isDirectory: true), alias] {
            XCTAssertThrowsError(try GLM5NextCheckpointConverter.validatedPaths(
                source: source,
                output: output)) { error in
                    XCTAssertTrue(error.localizedDescription.contains("filesystem or volume root"))
                }
        }
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: source.appendingPathComponent("config.json").path))
    }

    func testInspectRejectsMissingRequiredAssetBeforeCreatingOutput() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        try FileManager.default.removeItem(
            at: source.appendingPathComponent("processor_config.json"))

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: source,
            sourceRevision: String(repeating: "a", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("processor_config.json"))
            }
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testExplicitRevisionCannotContradictSnapshotPath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshotRevision = String(repeating: "a", count: 40)
        let snapshot = root.appendingPathComponent("snapshots/\(snapshotRevision)")
        let fixture = try makeFixture()
        try FileManager.default.createDirectory(
            at: snapshot.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: fixture, to: snapshot)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: snapshot,
            sourceRevision: String(repeating: "b", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("conflicts"))
            }
    }

    func testExplicitRevisionCannotContradictLocalMetadata() throws {
        let source = try makeFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let metadata = source.appendingPathComponent(
            ".cache/huggingface/download/config.json.metadata")
        try FileManager.default.createDirectory(
            at: metadata.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("\(String(repeating: "a", count: 40))\n".utf8).write(to: metadata)

        XCTAssertThrowsError(try GLM5NextCheckpointConverter.inspect(
            source: source,
            sourceRevision: String(repeating: "b", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("conflicts"))
            }
    }

    func testResumeRejectsSameSizeShardMutationWithPreservedModificationTime() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        let revision = String(repeating: "a", count: 40)
        let converter = GLM5NextCheckpointConverter(
            source: source, output: output, sourceRevision: revision)
        try converter.run()

        let shard = source.appendingPathComponent("model-00001-of-00002.safetensors")
        let modificationDate = try shard.resourceValues(
            forKeys: [.contentModificationDateKey]).contentModificationDate
        var bytes = try Data(contentsOf: shard)
        bytes[bytes.index(before: bytes.endIndex)] ^= 0xff
        try bytes.write(to: shard)
        if let modificationDate {
            try FileManager.default.setAttributes(
                [.modificationDate: modificationDate], ofItemAtPath: shard.path)
        }

        XCTAssertThrowsError(try converter.run()) { error in
            XCTAssertTrue(error.localizedDescription.contains("shard contents"))
        }
    }

    func testProviderResumeInspectionRejectsUnitsOutsidePlan() throws {
        let source = try makeFixture()
        let output = source.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: output)
        }
        let revision = String(repeating: "a", count: 40)
        try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: revision).run()

        let valid = try AFMMLXCheckpointConverter.inspectResume(
            source: source,
            output: output,
            sourceRevision: revision)
        XCTAssertGreaterThan(valid.verifiedCompletedOutputBytes, 0)

        let stateURL = output.appendingPathComponent(".afm-mlx-conversion.json")
        var state = try json(at: stateURL)
        var completed = try XCTUnwrap(state["completed"] as? [String: Any])
        completed["forged-unit"] = [
            "outputFile": "model-forged.safetensors",
            "outputSize": 1,
            "outputSHA256": String(repeating: "0", count: 64),
            "outputKeys": ["forged.weight"],
        ]
        state["completed"] = completed
        try JSONSerialization.data(withJSONObject: state, options: [.sortedKeys])
            .write(to: stateURL)

        XCTAssertThrowsError(try AFMMLXCheckpointConverter.inspectResume(
            source: source,
            output: output,
            sourceRevision: revision)) { error in
                XCTAssertTrue(error.localizedDescription.contains("outside the current conversion plan"))
            }
    }

    func testDispatcherPreservesDeepSeekProfilesAndDetectsGLM() throws {
        let source = try makeFixture()
        defer { try? FileManager.default.removeItem(at: source) }

        let inspection = try AFMMLXCheckpointConverter.inspect(
            source: source,
            sourceRevision: String(repeating: "a", count: 40))
        XCTAssertEqual(inspection.modelKind, .glm5Next)
        XCTAssertEqual(inspection.defaultProfile, "mlx-affine-4")
        XCTAssertEqual(inspection.sourceRevision, String(repeating: "a", count: 40))

        let deepSeek = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: deepSeek, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: deepSeek) }
        try Data("{\"model_type\":\"deepseek_v4\"}".utf8).write(
            to: deepSeek.appendingPathComponent("config.json"))
        let deepSeekInspection = try AFMMLXCheckpointConverter.inspect(source: deepSeek)
        XCTAssertEqual(deepSeekInspection.modelKind, .deepseekV4)
        XCTAssertEqual(
            deepSeekInspection.supportedProfiles,
            DeepseekV4CheckpointConverter.Profile.allCases.map(\.rawValue))
        XCTAssertThrowsError(try AFMMLXCheckpointConverter.inspect(
            source: deepSeek,
            sourceRevision: String(repeating: "b", count: 40))) { error in
                XCTAssertTrue(error.localizedDescription.contains("supported only for GLM-5.3"))
            }
    }

    private func makeFixture(
        rawExpertDType: String = "F8_E4M3",
        includeExpertScales: Bool = true,
        expertScaleDType: String = "F32"
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)

        let config: [String: Any] = [
            "model_type": "glm5_next",
            "text_config": [
                "model_type": "glm5_next_text",
                "num_hidden_layers": 2,
                "n_routed_experts": 12,
                "kv_lora_rank": 128,
                "num_attention_heads": 2,
                "qk_nope_head_dim": 64,
                "v_head_dim": 64,
                "num_nextn_predict_layers": 1,
            ],
            "vision_config": ["model_type": "glm5_next_vision"],
            "quantization_config": [
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            ],
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("config.json"))

        let expertShape = [128, 128]
        let expertBytes = expertShape.reduce(1, *)
        let scale = floatData([1])
        var shardOne = [String: FixtureTensor]()
        var shardTwo = [String: FixtureTensor]()
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for expert in 0..<12 {
                let base = "model.language_model.layers.1.mlp.experts.\(expert).\(projection)"
                let raw: UInt8 = expert == 2 ? 0x40 : (expert == 10 ? 0x48 : 0x38)
                var destination = expert.isMultiple(of: 2) ? shardOne : shardTwo
                destination["\(base).weight"] = FixtureTensor(
                    dtype: rawExpertDType,
                    shape: expertShape,
                    data: Data(repeating: raw, count: expertBytes))
                if includeExpertScales {
                    destination["\(base).weight_scale_inv"] = FixtureTensor(
                        dtype: expertScaleDType,
                        shape: [1, 1],
                        data: expertScaleDType == "F32" ? scale : Data(repeating: 0, count: 2))
                }
                if expert.isMultiple(of: 2) {
                    shardOne = destination
                } else {
                    shardTwo = destination
                }
            }
        }

        let convolutionShape = [128, 2, 1]
        for component in ["q", "k", "v"] {
            shardOne[
                "model.language_model.layers.0.self_attn.\(component)_conv1d.weight"
            ] = FixtureTensor(
                dtype: "BF16",
                shape: convolutionShape,
                data: Data(repeating: 0, count: convolutionShape.reduce(1, *) * 2))
        }
        let kvShape = [256, 128]
        shardTwo["model.language_model.layers.1.self_attn.kv_b_proj.weight"] =
            FixtureTensor(
                dtype: "BF16",
                shape: kvShape,
                data: Data(repeating: 0, count: kvShape.reduce(1, *) * 2))
        shardTwo["model.visual.patch_embed.proj.weight"] = FixtureTensor(
            dtype: "BF16",
            shape: [1024, 3, 2, 14, 14],
            data: Data(repeating: 0, count: 1024 * 3 * 2 * 14 * 14 * 2))
        shardTwo["model.language_model.layers.2.mtp_projection.weight"] = FixtureTensor(
            dtype: "BF16",
            shape: [4, 4],
            data: Data(repeating: 0, count: 4 * 4 * 2))

        let shardOneName = "model-00001-of-00002.safetensors"
        let shardTwoName = "model-00002-of-00002.safetensors"
        try writeSafetensor(shardOne, to: root.appendingPathComponent(shardOneName))
        try writeSafetensor(shardTwo, to: root.appendingPathComponent(shardTwoName))
        var weightMap = [String: String]()
        for name in shardOne.keys { weightMap[name] = shardOneName }
        for name in shardTwo.keys { weightMap[name] = shardTwoName }
        let index: [String: Any] = [
            "metadata": ["total_size": shardOne.values.reduce(0) { $0 + $1.data.count }
                + shardTwo.values.reduce(0) { $0 + $1.data.count }],
            "weight_map": weightMap,
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))

        let assets: [String: String] = [
            "chat_template.jinja": "{{ messages }}",
            "processor_config.json": "{}",
            "tokenizer.json": "{}",
            "tokenizer_config.json": "{}",
        ]
        for (name, contents) in assets {
            try Data(contents.utf8).write(to: root.appendingPathComponent(name))
        }
        return root
    }

    private func writeSafetensor(
        _ tensors: [String: FixtureTensor],
        to url: URL
    ) throws {
        var header = [String: Any]()
        var payload = Data()
        var offset = 0
        for name in tensors.keys.sorted() {
            let tensor = tensors[name]!
            header[name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + tensor.data.count],
            ]
            payload.append(tensor.data)
            offset += tensor.data.count
        }
        var headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        while !headerData.count.isMultiple(of: 8) { headerData.append(0x20) }
        var headerSize = UInt64(headerData.count).littleEndian
        var file = withUnsafeBytes(of: &headerSize) { Data($0) }
        file.append(headerData)
        file.append(payload)
        try file.write(to: url)
    }

    private func floatData(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func json(at url: URL) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    private func assertSourceMismatch(source: URL, output: URL, revision: String) {
        XCTAssertThrowsError(try GLM5NextCheckpointConverter(
            source: source,
            output: output,
            sourceRevision: revision).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains(
                    "Source config, index, shard contents, or support assets changed"))
            }
    }
}
