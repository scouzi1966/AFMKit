import Foundation
import MLX
import XCTest

@testable import AFMKitMLX

final class Qwen4ExpCheckpointConverterTests: XCTestCase {
    private let revision = String(repeating: "a", count: 40)

    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    func testInspectionRecognizesOfficialQwenNextLayout() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let inspection = try Qwen4ExpCheckpointConverter.inspect(
            source: fixture.source,
            sourceRevision: revision)

        XCTAssertEqual(inspection.sourceRevision, revision)
        XCTAssertEqual(inspection.shardCount, 1)
        XCTAssertEqual(inspection.ngramShardCount, 128)
        XCTAssertGreaterThan(inspection.sourceBytes, 0)
        XCTAssertGreaterThan(inspection.estimatedOutputBytes, 0)
    }

    func testConversionProducesMappedSidecarAndRuntimeWeightLayout() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        try Qwen4ExpCheckpointConverter(
            source: fixture.source,
            output: fixture.output,
            sourceRevision: revision).run()

        let config = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: fixture.output.appendingPathComponent("config.json")))
                as? [String: Any])
        let descriptor = try XCTUnwrap(config["ngram_table"] as? [String: Any])
        XCTAssertEqual(descriptor["file"] as? String, "ngram_table.ngram")
        XCTAssertEqual(descriptor["bits"] as? Int, 4)
        XCTAssertEqual(descriptor["group_size"] as? Int, 32)

        let sidecar = fixture.output.appendingPathComponent("ngram_table.ngram")
        let sidecarHeader = try AFMSafetensorHeader(url: sidecar)
        let sidecarMetadata = try sidecarMetadata(at: sidecar)
        XCTAssertEqual(sidecarMetadata["format"], "mlx-serve-ngram")
        XCTAssertEqual(sidecarMetadata["bits"], "4")
        XCTAssertEqual(sidecarMetadata["group_size"], "32")
        let byName = Dictionary(uniqueKeysWithValues: sidecarHeader.tensors.map { ($0.name, $0) })
        XCTAssertEqual(byName["weight"]?.dtype, .uint32)
        XCTAssertEqual(byName["weight"]?.shape, [128, 20])
        XCTAssertEqual(byName["scales"]?.shape, [128, 5])
        XCTAssertEqual(byName["biases"]?.shape, [128, 5])

        let index = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: fixture.output.appendingPathComponent("model.safetensors.index.json")))
                as? [String: Any])
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        XCTAssertNil(weightMap.keys.first(where: { $0.contains("ngram_embedding") }))
        XCTAssertNotNil(weightMap["language_model.lm_head.weight"])
        XCTAssertNotNil(weightMap["language_model.lm_head.scales"])
        XCTAssertNotNil(weightMap["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertNotNil(weightMap["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"])
        XCTAssertNotNil(weightMap["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"])

        let shard = try XCTUnwrap(weightMap["language_model.lm_head.weight"])
        let arrays = try loadArrays(url: fixture.output.appendingPathComponent(shard))
        XCTAssertEqual(arrays["language_model.lm_head.weight"]?.shape, [32, 16])
        XCTAssertEqual(arrays["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(
            arrays["language_model.model.layers.0.self_attn.conv1d.weight"]?.shape,
            [64, 4, 1])

        // A completed conversion is resumable and does not rewrite valid units.
        let before = try Data(contentsOf: sidecar)
        try Qwen4ExpCheckpointConverter(
            source: fixture.source,
            output: fixture.output,
            sourceRevision: revision).run()
        XCTAssertEqual(try Data(contentsOf: sidecar), before)
    }

    func testDispatcherAdvertisesMappedProfile() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let inspection = try AFMMLXCheckpointConverter.inspect(
            source: fixture.source,
            sourceRevision: revision)

        XCTAssertEqual(inspection.modelKind, .qwen4Exp)
        XCTAssertEqual(inspection.defaultProfile, "afm-mapped-4")
        XCTAssertEqual(inspection.supportedProfiles, ["afm-mapped-4"])
    }

    func testRejectsMissingNGramShards() throws {
        let fixture = try makeFixture(ngramCount: 127)
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        XCTAssertThrowsError(
            try Qwen4ExpCheckpointConverter.inspect(
                source: fixture.source,
                sourceRevision: revision)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Expected 128"))
        }
    }

    private func makeFixture(ngramCount: Int = 128) throws -> (
        root: URL, source: URL, output: URL
    ) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-qwen-converter-\(UUID().uuidString)")
        let source = root.appendingPathComponent("source")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let config: [String: Any] = [
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": [
                "model_type": "qwen4_exp_text",
                "split_ngram_parts": 128,
            ],
            "vision_config": ["model_type": "qwen4_exp"],
        ]
        try JSONSerialization.data(
            withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: source.appendingPathComponent("config.json"))

        var arrays = [String: MLXArray]()
        for index in 0..<ngramCount {
            arrays[
                "model.language_model.ple.ple_embedding.ngram_embedding.shard_\(index).weight"
            ] = MLXArray(Array(repeating: Float(index % 7) / 8, count: 160))
                .reshaped([1, 160]).asType(.bfloat16)
        }
        arrays["lm_head.weight"] = constant(rows: 32, columns: 64, value: 0.25)
        arrays["model.language_model.embed_tokens.weight"] = constant(
            rows: 32, columns: 64, value: 0.5)
        arrays["model.language_model.layers.0.mlp.experts.gate_up_proj"] = MLXArray(
            Array(repeating: Float(0.125), count: 2 * 128 * 64))
            .reshaped([2, 128, 64]).asType(.bfloat16)
        arrays["model.language_model.layers.0.mlp.experts.down_proj"] = MLXArray(
            Array(repeating: Float(0.375), count: 2 * 64 * 64))
            .reshaped([2, 64, 64]).asType(.bfloat16)
        arrays["model.language_model.layers.0.self_attn.conv1d.weight"] = MLXArray(
            Array(repeating: Float(0.75), count: 64 * 4))
            .reshaped([64, 1, 4]).asType(.bfloat16)
        arrays["model.language_model.layers.0.self_attn.q_norm.weight"] = MLXArray(
            Array(repeating: Float(0.1), count: 64)).asType(.bfloat16)
        arrays["model.visual.probe"] = MLXArray([Float(1)]).asType(.bfloat16)
        arrays["mtp.probe"] = MLXArray([Float(2)]).asType(.bfloat16)

        let shard = "model-00001-of-00001.safetensors"
        try save(arrays: arrays, url: source.appendingPathComponent(shard))
        let index: [String: Any] = [
            "metadata": ["total_size": arrays.values.reduce(0) { $0 + $1.nbytes }],
            "weight_map": Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, shard) }),
        ]
        try JSONSerialization.data(
            withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            .write(to: source.appendingPathComponent("model.safetensors.index.json"))
        for name in [
            "chat_template.jinja", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json",
        ] {
            try Data("{}".utf8).write(to: source.appendingPathComponent(name))
        }
        return (root, source, output)
    }

    private func constant(rows: Int, columns: Int, value: Float) -> MLXArray {
        MLXArray(Array(repeating: value, count: rows * columns))
            .reshaped([rows, columns]).asType(.bfloat16)
    }

    private func sidecarMetadata(at url: URL) throws -> [String: String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try XCTUnwrap(try handle.read(upToCount: 8))
        XCTAssertEqual(prefix.count, 8)
        let length = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        let header = try XCTUnwrap(try handle.read(upToCount: Int(length)))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: header) as? [String: Any])
        return try XCTUnwrap(object["__metadata__"] as? [String: String])
    }
}
