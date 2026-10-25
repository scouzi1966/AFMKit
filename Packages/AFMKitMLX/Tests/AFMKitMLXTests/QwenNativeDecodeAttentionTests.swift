import CryptoKit
import Foundation
import MLX
import MLXFast
import MLXNN
@testable import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenNativeDecodeAttentionTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String?
        let layer: Int?
        let afm_stock_attention: Bool?
        let key_length: Int?
        let expect_native_mask: Bool?
        let native_capture: String
        let afm_capture: String
        let files: [String: String]
    }

    func testOptionalCapturedDecodeAttention() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_DECODE_ATTN_FIXTURE"],
              let output = env["AFM_QWEN_DECODE_ATTN_OUT"] else {
            throw XCTSkip("Requires qualified first-decode attention captures")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = true
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertFalse(fixture.files.isEmpty)
        for (file, expected) in fixture.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: file))
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected, file)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let native = try loadArrays(url: URL(fileURLWithPath: fixture.native_capture))
        let afm = try loadArrays(url: URL(fileURLWithPath: fixture.afm_capture))
        let q = try XCTUnwrap(native["attention-query"])
        let k = try XCTUnwrap(native["attention-cached-key"])
        let v = try XCTUnwrap(native["attention-cached-value"])
        let expected = try XCTUnwrap(native["attention-heads"])
        let captured = try XCTUnwrap(afm["attention-heads"])
        let mask = try XCTUnwrap(native["qsa-selection"])
        XCTAssertEqual(q.shape, [1, 24, 1, 256])
        let keyLength = fixture.key_length ?? 6662
        let completeBlocks = keyLength / 4
        XCTAssertEqual(k.shape, [1, 2, keyLength, 256])
        XCTAssertEqual(v.shape, k.shape)
        let visible = mask.asArray(Bool.self)
        XCTAssertEqual(visible.count, keyLength)
        var selected = [Int32]()
        for block in 0..<completeBlocks {
            let group = Array(visible[(block * 4)..<(block * 4 + 4)])
            XCTAssertTrue(group.allSatisfy { $0 == group[0] })
            if group[0] { selected.append(Int32(block)) }
        }
        XCTAssertEqual(selected.count, 512)
        XCTAssertTrue(visible.suffix(keyLength % 4).allSatisfy { $0 })
        let nativeBlocks = MLXArray(selected).reshaped(1, 1, 512)
        let scores = try XCTUnwrap(Qwen4ExpQSADecodeScores.call(
            queries: try XCTUnwrap(afm["qsa-queries"]),
            blockKeys: try XCTUnwrap(afm["qsa-pooled"])))
        let afmBlocks = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
            scores: scores.expandedDimensions(axis: 0),
            visibleBlockCounts: [completeBlocks], topK: 512)).reshaped(1, 1, 512)
        let afmMask = Qwen4ExpQSAGather.maskFromBlocks(
            afmBlocks, keyLength: keyLength, compressionRatio: 4)
        let rankedMask = try XCTUnwrap(Qwen4ExpQSADecodeMask.call(
            scores: scores, visibleCount: keyLength, keyLength: keyLength,
            compressionRatio: 4, blockTopK: 512))
        func stock(_ selectedMask: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 0.0625, mask: .array(selectedMask))
        }
        func direct(_ blocks: MLXArray) throws -> MLXArray {
            try XCTUnwrap(Qwen4ExpQSADecodeAttention.call(queries: q, keys: k, values: v,
                scale: 0.0625, selectedBlocks: blocks, compressionRatio: 4))
        }
        let stockNative = stock(mask), stockAFM = stock(afmMask)
        let stockRanked = stock(rankedMask)
        let directNative = try direct(nativeBlocks), directAFM = try direct(afmBlocks)
        var records = [[String: Any]]()
        func record(_ name: String, _ actual: MLXArray, _ reference: MLXArray) {
            XCTAssertEqual(actual.shape, reference.shape, name)
            let difference = actual.asType(.float32) - reference.asType(.float32)
            let count = (actual .!= reference).asType(.int32).sum().item(Int.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["stage": name, "different": count, "values": actual.size,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("DECODE_ATTN stage=\(name) different=\(count)/\(actual.size) rms=\(rms)")
        }
        record("selected-blocks", afmBlocks, nativeBlocks)
        record("selected-mask", afmMask, mask)
        record("ranked-mask-vs-native", rankedMask, mask)
        record("stock-ranked-mask-vs-native", stockRanked, expected)
        record("stock-ranked-mask-vs-captured-afm", stockRanked, captured)
        record("stock-native-mask-vs-native", stockNative, expected)
        record("stock-afm-mask-vs-native", stockAFM, expected)
        record("direct-native-blocks-vs-native", directNative, expected)
        record("direct-afm-blocks-vs-native", directAFM, expected)
        record("direct-afm-blocks-vs-captured-afm", directAFM, captured)
        let reconstruction = fixture.afm_stock_attention == true ? stockRanked : directAFM
        let reconstructionTarget = fixture.expect_native_mask == true ? expected : captured
        let reconstructionMatches = reconstruction.asData(access: .copy).data == reconstructionTarget.asData(access: .copy).data
        let selectionMatches = rankedMask.asData(access: .copy).data == mask.asData(access: .copy).data
        var tailArrays = [String: MLXArray]()
        if let directory = fixture.model, let layer = fixture.layer {
            let context = try await LLMModelFactory.shared.load(configuration:
                ModelConfiguration(directory: URL(fileURLWithPath: directory)))
            let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
            let projection = try XCTUnwrap(model.namedModules().first {
                $0.0 == "model.layers.\(layer).self_attn.o_proj"
            }?.1 as? Linear)
            let gate = try XCTUnwrap(native["attention-gate"]).reshaped(1, 1, -1)
            let flat = expected.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
            let ordinaryGate = sigmoid(gate)
            let preciseGate = Qwen4ExpHCActivation.sigmoid(gate)
            let ordinaryTail = compile(shapeless: false) { (arrays: [MLXArray]) -> [MLXArray] in
                CompiledDecodeTrace.withActive {
                    [qwen4ExpVerificationLinear(projection, arrays[0] * sigmoid(arrays[1]),
                        verificationPolicy: nil, role: .attention)]
                }
            }
            let preciseTail = compile(shapeless: false) { (arrays: [MLXArray]) -> [MLXArray] in
                CompiledDecodeTrace.withActive {
                    [qwen4ExpVerificationLinear(projection, arrays[0] * Qwen4ExpHCActivation.sigmoid(arrays[1]),
                        verificationPolicy: nil, role: .attention)]
                }
            }
            let ordinary = ordinaryTail([flat, gate])[0]
            let precise = preciseTail([flat, gate])[0]
            let key = String(format: "layer-%03d-attn-output", layer)
            let nativeOutput = try XCTUnwrap(native[key])
            let afmOutput = try XCTUnwrap(afm[key])
            record("output-gate-ordinary-vs-precise", ordinaryGate, preciseGate)
            record("output-tail-ordinary-vs-native", ordinary, nativeOutput)
            record("output-tail-ordinary-vs-captured-afm", ordinary, afmOutput)
            record("output-tail-precise-vs-native", precise, nativeOutput)
            tailArrays = ["ordinary-gate": ordinaryGate, "precise-gate": preciseGate,
                "ordinary-tail": ordinary, "precise-tail": precise]
        }
        for _ in 0..<8 { eval(stock(mask)); eval(try direct(nativeBlocks)) }
        var timings = [[String: Any]]()
        for sample in 0..<30 {
            for name in sample.isMultiple(of: 2) ? ["stock-mask", "direct-blocks"] : ["direct-blocks", "stock-mask"] {
                let start = DispatchTime.now().uptimeNanoseconds
                if name == "stock-mask" { eval(stock(mask)) } else { eval(try direct(nativeBlocks)) }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                timings.append(["sample": sample, "implementation": name, "milliseconds": ms])
            }
        }
        for sample in 0..<34 {
            for name in sample.isMultiple(of: 2) ? ["rank-mask", "radix-mask"] : ["radix-mask", "rank-mask"] {
                let start = DispatchTime.now().uptimeNanoseconds
                if name == "rank-mask" {
                    eval(try XCTUnwrap(Qwen4ExpQSADecodeMask.call(
                        scores: scores, visibleCount: keyLength, keyLength: keyLength,
                        compressionRatio: 4, blockTopK: 512)))
                } else {
                    let blocks = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                        scores: scores.expandedDimensions(axis: 0),
                        visibleBlockCounts: [completeBlocks], topK: 512))
                    eval(Qwen4ExpQSAGather.maskFromBlocks(
                        blocks, keyLength: keyLength, compressionRatio: 4))
                }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                if sample >= 4 {
                    timings.append(["sample": sample, "implementation": name, "milliseconds": ms])
                }
            }
        }
        try save(arrays: ["stock-native": stockNative, "stock-afm": stockAFM,
            "direct-native": directNative, "direct-afm": directAFM,
            "native-blocks": nativeBlocks, "afm-blocks": afmBlocks, "scores": scores,
            "native-mask": mask, "afm-mask": afmMask, "ranked-mask": rankedMask,
            "stock-ranked": stockRanked].merging(tailArrays) { _, n in n }, url: root.appendingPathComponent("stages.safetensors"))
        let report: [String: Any] = ["records": records, "timings": timings,
            "scope": "Identical captured first-decode operands. Alternating component timings include graph construction and evaluation; not model throughput or coding quality."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        XCTAssertTrue(reconstructionMatches,
                      "Reconstruction must reproduce the captured ordinary AFM attention")
        if fixture.expect_native_mask == true { XCTAssertTrue(selectionMatches) }
        #endif
    }
}
