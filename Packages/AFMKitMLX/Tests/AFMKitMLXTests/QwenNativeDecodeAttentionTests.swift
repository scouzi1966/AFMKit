import CryptoKit
import Foundation
import MLX
import MLXFast
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenNativeDecodeAttentionTests: XCTestCase {
    private struct Fixture: Decodable {
        let native_capture: String
        let afm_capture: String
        let files: [String: String]
    }

    func testOptionalCapturedDecodeAttention() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_DECODE_ATTN_FIXTURE"],
              let output = env["AFM_QWEN_DECODE_ATTN_OUT"] else {
            throw XCTSkip("Requires qualified first-decode attention captures")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
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
        XCTAssertEqual(k.shape, [1, 2, 6662, 256])
        XCTAssertEqual(v.shape, k.shape)
        let visible = mask.asArray(Bool.self)
        XCTAssertEqual(visible.count, 6662)
        var selected = [Int32]()
        for block in 0..<1665 {
            let group = Array(visible[(block * 4)..<(block * 4 + 4)])
            XCTAssertTrue(group.allSatisfy { $0 == group[0] })
            if group[0] { selected.append(Int32(block)) }
        }
        XCTAssertEqual(selected.count, 512)
        XCTAssertTrue(visible.suffix(2).allSatisfy { $0 })
        let nativeBlocks = MLXArray(selected).reshaped(1, 1, 512)
        let scores = try XCTUnwrap(Qwen4ExpQSADecodeScores.call(
            queries: try XCTUnwrap(afm["qsa-queries"]),
            blockKeys: try XCTUnwrap(afm["qsa-pooled"])))
        let afmBlocks = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
            scores: scores.expandedDimensions(axis: 0),
            visibleBlockCounts: [1665], topK: 512)).reshaped(1, 1, 512)
        let afmMask = Qwen4ExpQSAGather.maskFromBlocks(
            afmBlocks, keyLength: 6662, compressionRatio: 4)
        func stock(_ selectedMask: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                scale: 0.0625, mask: .array(selectedMask))
        }
        func direct(_ blocks: MLXArray) throws -> MLXArray {
            try XCTUnwrap(Qwen4ExpQSADecodeAttention.call(queries: q, keys: k, values: v,
                scale: 0.0625, selectedBlocks: blocks, compressionRatio: 4))
        }
        let stockNative = stock(mask), stockAFM = stock(afmMask)
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
        record("stock-native-mask-vs-native", stockNative, expected)
        record("stock-afm-mask-vs-native", stockAFM, expected)
        record("direct-native-blocks-vs-native", directNative, expected)
        record("direct-afm-blocks-vs-native", directAFM, expected)
        record("direct-afm-blocks-vs-captured-afm", directAFM, captured)
        XCTAssertEqual(directAFM.asData(access: .copy).data, captured.asData(access: .copy).data,
                       "Reconstruction must reproduce the captured ordinary AFM attention")
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
        try save(arrays: ["stock-native": stockNative, "stock-afm": stockAFM,
            "direct-native": directNative, "direct-afm": directAFM,
            "native-blocks": nativeBlocks, "afm-blocks": afmBlocks, "scores": scores,
            "native-mask": mask, "afm-mask": afmMask], url: root.appendingPathComponent("stages.safetensors"))
        let report: [String: Any] = ["records": records, "timings": timings,
            "scope": "Identical first-decode layer-3 operands. Alternating component timings include graph construction and evaluation; not model throughput or coding quality."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        #endif
    }
}
