import Foundation
import CryptoKit
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Explicit local-checkpoint gate. Direct model timings are NOT API results.
final class QwenOnePassPromptNativeTests: XCTestCase {
    private func cacheArrays(_ cache: KVCache) -> [MLXArray?] {
        if let cache = cache as? Qwen4ExpLayerCache {
            return (0..<4).map { cache[$0] }
        }
        return (cache as! Qwen4ExpAttentionCache).promptReplayArraysForTesting
    }

    func testOptionalNativeCaptureAndReplay() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_ONE_PASS_MODEL"],
              let fixturePath = env["AFM_TEST_ONE_PASS_FIXTURE"],
              let reportPath = env["AFM_TEST_ONE_PASS_REPORT"] else {
            throw XCTSkip("Explicit native checkpoint, saved Context prompts and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only qualification")
        #else
        continueAfterFailure = false
        struct Prompt: Decodable { let label: String; let prompt: String }
        let reportURL = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard reportURL.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: reportURL.path) else {
            throw NSError(domain: "OnePassNative", code: 1)
        }
        let fixtures = try JSONDecoder().decode([Prompt].self,
            from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        XCTAssertEqual(fixtures.count, 4)
        let suffixDiagnosticOnly = env["AFM_TEST_ONE_PASS_SUFFIX_ONLY"] == "1"
        let selectedLabel = env["AFM_TEST_ONE_PASS_SELECTED_LABEL"]
        let selectedFixtures = fixtures.filter { selectedLabel == nil || $0.label == selectedLabel }
        XCTAssertFalse(selectedFixtures.isEmpty)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let directory = URL(fileURLWithPath: modelPath)
        let config = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any])
        let quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
        let groupSize = try XCTUnwrap(quantization["group_size"] as? Int)
        let bits = try XCTUnwrap(quantization["bits"] as? Int)
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: directory))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let head = try model.loadEmbeddedMTPHead(
            modelDirectory: directory, groupSize: groupSize, bits: bits)
        let control = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .strictSingletonEquivalent, draftDispatchStride: 0)
        let candidate = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .strictSingletonEquivalent, draftDispatchStride: 0, onePassPromptCapture: true)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        var records = [[String: Any]]()
        func write() throws {
            try JSONSerialization.data(withJSONObject: [
                "scope": "Direct native text-model/MTP gate; not HTTP/VLM, aggregate throughput, or semantic quality qualification. All four lengths force retention, including >4096 where server admission normally declines.",
                "model": modelPath, "prefill_step": step, "temperature": 0, "top_p": 1,
                "depth": 3, "records": records,
            ], options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
        }
        func drain(_ session: Qwen4ExpMTPSession) -> (tokens: [Int], milliseconds: Double) {
            let start = DispatchTime.now().uptimeNanoseconds
            var tokens = [Int]()
            while let token = session.nextToken() { tokens.append(token) }
            return (tokens, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        func hash(_ tokens: [Int]) -> String {
            let data = tokens.map(Int32.init).withUnsafeBufferPointer { Data(buffer: $0) }
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        for fixture in selectedFixtures.prefix(suffixDiagnosticOnly ? 1 : selectedFixtures.count) {
            let ids = try context.tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": fixture.prompt]], tools: nil,
                additionalContext: ["enable_thinking": false])
            for round in 0..<(suffixDiagnosticOnly ? 1 : 4) {
                let order = ["endpoint", "split", "one-pass"]
                var sessions = [String: Qwen4ExpMTPSession]()
                var snapshots = [String: Qwen4ExpMTPPromptState]()
                var timing = [String: Double]()
                for i in 0..<3 {
                    let arm = order[(i + round) % order.count]
                    let generator = arm == "one-pass" ? candidate : control
                    let start = DispatchTime.now().uptimeNanoseconds
                    let session = try XCTUnwrap(generator.makeSession(promptIds: ids, maxTokens: 128,
                        retainPromptState: true, prefillStepSize: step,
                        promptSnapshotBackoffTokens: arm == "endpoint" ? 0 : 31))
                    timing[arm] = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    sessions[arm] = session
                    snapshots[arm] = try XCTUnwrap(session.takePromptState())
                }
                if fixture.label == "0.5K", round == 0 {
                    let splitTarget = model.newCache(parameters: nil)
                    let onePassTarget = model.newCache(parameters: nil)
                    let splitHead = head.newCache()
                    let onePassHead = head.newCache()
                    let splitBridge = snapshots["split"]!.restoreForTesting(
                        target: splitTarget, head: splitHead)
                    let onePassBridge = snapshots["one-pass"]!.restoreForTesting(
                        target: onePassTarget, head: onePassHead)
                    var differences = [[String: Any]]()
                    for (name, pair) in [
                        ("hidden", (splitBridge.hidden, onePassBridge.hidden)),
                        ("stream", (splitBridge.stream, onePassBridge.stream)),
                    ] where !arrayEqual(pair.0, pair.1).item(Bool.self) {
                        differences.append(["state": name])
                    }
                    for (layer, pair) in zip(splitTarget + splitHead,
                                              onePassTarget + onePassHead).enumerated() {
                        if pair.0.offset != pair.1.offset {
                            differences.append(["layer": layer, "offset_split": pair.0.offset,
                                "offset_one_pass": pair.1.offset])
                        }
                        let left = cacheArrays(pair.0), right = cacheArrays(pair.1)
                        for slot in left.indices {
                            switch (left[slot], right[slot]) {
                            case (.none, .none): break
                            case (.some(let x), .some(let y)):
                                let sameShape = x.shape == y.shape
                                let sameType = x.dtype == y.dtype
                                let sameValues = sameShape && sameType
                                    && arrayEqual(x, y).item(Bool.self)
                                if !sameValues {
                                    differences.append(["layer": layer, "slot": slot,
                                        "shape": x.shape, "dtype": "\(x.dtype)",
                                        "shape_or_type_mismatch": !sameShape || !sameType])
                                }
                            default:
                                differences.append(["layer": layer, "slot": slot,
                                    "shape_or_presence_mismatch": true])
                            }
                        }
                    }
                    records.append(["label": fixture.label, "round": round,
                        "kind": "split-versus-one-pass-boundary", "differences": differences])
                    // A fixed full-width prefill with a changed suffix isolates
                    // snapshot causality from split-versus-full width arithmetic.
                    let boundary = ids.count - 31
                    var alternateIDs = ids
                    for index in boundary..<alternateIDs.count {
                        alternateIDs[index] = alternateIDs[index] == 3 ? 4 : 3
                    }
                    let alternate = try XCTUnwrap(candidate.makeSession(
                        promptIds: alternateIDs, maxTokens: 128,
                        retainPromptState: true, prefillStepSize: step,
                        promptSnapshotBackoffTokens: 31))
                    let alternateSnapshot = try XCTUnwrap(alternate.takePromptState())
                    XCTAssertEqual(alternateSnapshot.promptIds, snapshots["one-pass"]!.promptIds)
                    let alternateTarget = model.newCache(parameters: nil)
                    let alternateHead = head.newCache()
                    let alternateBridge = alternateSnapshot.restoreForTesting(
                        target: alternateTarget, head: alternateHead)
                    var suffixDifferences = [[String: Any]]()
                    if !arrayEqual(onePassBridge.hidden, alternateBridge.hidden).item(Bool.self) {
                        suffixDifferences.append(["state": "hidden"])
                    }
                    if !arrayEqual(onePassBridge.stream, alternateBridge.stream).item(Bool.self) {
                        suffixDifferences.append(["state": "stream"])
                    }
                    for (layer, pair) in zip(onePassTarget + onePassHead,
                                              alternateTarget + alternateHead).enumerated() {
                        if pair.0.offset != pair.1.offset {
                            suffixDifferences.append(["layer": layer, "state": "offset"])
                        }
                        let left = cacheArrays(pair.0), right = cacheArrays(pair.1)
                        for slot in left.indices {
                            switch (left[slot], right[slot]) {
                            case (.none, .none): break
                            case (.some(let x), .some(let y)):
                                if x.shape != y.shape || x.dtype != y.dtype
                                    || !arrayEqual(x, y).item(Bool.self) {
                                    suffixDifferences.append(["layer": layer, "slot": slot])
                                }
                            default:
                                suffixDifferences.append(["layer": layer, "slot": slot,
                                    "state": "presence"])
                            }
                        }
                    }
                    records.append(["label": fixture.label, "round": round,
                        "kind": "one-pass-suffix-isolation", "differences": suffixDifferences])
                    let fullIDs = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
                    let prefixIDs = MLXArray(Array(ids.prefix(boundary)).map(Int32.init)).reshaped(1, -1)
                    let fullPLE = try XCTUnwrap(model.firstPLETraceForTesting(inputIDs: fullIDs))
                    let prefixPLE = try XCTUnwrap(model.firstPLETraceForTesting(inputIDs: prefixIDs))
                    let fullEmbeddingPrefix = fullPLE.embedding[0..., ..<boundary, 0...]
                    let fullOutputPrefix = fullPLE.output[0..., ..<boundary, 0...]
                    records.append(["label": fixture.label, "round": round,
                        "kind": "first-ple-prefix-width", "boundary": boundary,
                        "embedding_exact": arrayEqual(fullEmbeddingPrefix, prefixPLE.embedding).item(Bool.self),
                        "output_exact": arrayEqual(fullOutputPrefix, prefixPLE.output).item(Bool.self),
                        "embedding_max_error": abs(fullEmbeddingPrefix.asType(.float32)
                            - prefixPLE.embedding.asType(.float32)).max().item(Float.self),
                        "output_max_error": abs(fullOutputPrefix.asType(.float32)
                            - prefixPLE.output.asType(.float32)).max().item(Float.self)])
                    // The checkpoint's first PLE is in decoder 1. Inspect the
                    // native decoder 0 boundary before attributing any later
                    // first-PLE output difference to PLE arithmetic.
                    let fullStreams = model.layerStreamsForTesting(
                        inputIDs: fullIDs, cache: model.newCache(parameters: nil))
                    let prefixStreams = model.layerStreamsForTesting(
                        inputIDs: prefixIDs, cache: model.newCache(parameters: nil))
                    for index in 0..<min(3, fullStreams.count, prefixStreams.count) {
                        let fullPrefix = fullStreams[index][0..., ..<boundary]
                        let prefix = prefixStreams[index]
                        XCTAssertEqual(fullPrefix.shape, prefix.shape)
                        records.append(["label": fixture.label, "round": round,
                            "kind": "native-layer-prefix-width", "boundary": boundary,
                            "stage": index == 0 ? "embedding"
                                : (index == fullStreams.count - 1 ? "final_mixer"
                                    : "decoder_\(index - 1)"),
                            "exact": arrayEqual(fullPrefix, prefix).item(Bool.self),
                            "max_error": abs(fullPrefix.asType(.float32)
                                - prefix.asType(.float32)).max().item(Float.self)])
                    }
                    let fullLayerStages = model.firstLinearLayerStageTraceForTesting(
                        inputIDs: fullIDs, cache: model.newCache(parameters: nil))
                    let prefixLayerStages = model.firstLinearLayerStageTraceForTesting(
                        inputIDs: prefixIDs, cache: model.newCache(parameters: nil))
                    XCTAssertEqual(fullLayerStages.map(\.0), prefixLayerStages.map(\.0))
                    XCTAssertTrue(arrayEqual(fullLayerStages.last!.1, fullStreams[1]).item(Bool.self),
                        "full-width layer trace must reproduce normal decoder 0")
                    XCTAssertTrue(arrayEqual(prefixLayerStages.last!.1, prefixStreams[1]).item(Bool.self),
                        "prefix-only layer trace must reproduce normal decoder 0")
                    let moeIndex = try XCTUnwrap(fullLayerStages.firstIndex { $0.0 == "moe_output" })
                    XCTAssertTrue(arrayEqual(fullLayerStages[moeIndex].1,
                        fullLayerStages[moeIndex + 1].1).item(Bool.self),
                        "full-width MoE trace must reproduce normal MoE output")
                    XCTAssertTrue(arrayEqual(prefixLayerStages[moeIndex].1,
                        prefixLayerStages[moeIndex + 1].1).item(Bool.self),
                        "prefix-only MoE trace must reproduce normal MoE output")
                    for (full, short) in zip(fullLayerStages, prefixLayerStages) {
                        let fullPrefix = full.1[0..., ..<boundary]
                        XCTAssertEqual(fullPrefix.shape, short.1.shape)
                        var row: [String: Any] = ["label": fixture.label, "round": round,
                            "kind": "native-decoder0-stage-prefix-width", "boundary": boundary,
                            "stage": full.0,
                            "exact": arrayEqual(fullPrefix, short.1).item(Bool.self),
                            "max_error": abs(fullPrefix.asType(.float32)
                                - short.1.asType(.float32)).max().item(Float.self)]
                        if full.0 == "router_logits" || full.0 == "route_indices" {
                            let different = (fullPrefix .!= short.1).asType(.int32)
                            let perRow = different.sum(axis: -1).reshaped(-1).asArray(Int32.self)
                            row["different_elements"] = different.sum().item(Int.self)
                            row["different_rows"] = perRow.filter { $0 != 0 }.count
                            row["first_different_row"] = perRow.firstIndex { $0 != 0 } ?? -1
                            row["last_different_row"] = perRow.lastIndex { $0 != 0 } ?? -1
                        }
                        records.append(row)
                    }
                    // M=483 and M=514 straddle the quantized router's
                    // split-K dispatch threshold. Pad only the prefix router
                    // input above 512 to test that source-derived hypothesis.
                    let paddedStages = model.firstLinearLayerStageTraceForTesting(
                        inputIDs: prefixIDs, cache: model.newCache(parameters: nil),
                        routerPadTo: 513)
                    XCTAssertEqual(fullLayerStages.map(\.0), paddedStages.map(\.0))
                    for (full, padded) in zip(fullLayerStages, paddedStages) {
                        let fullPrefix = full.1[0..., ..<boundary]
                        XCTAssertEqual(fullPrefix.shape, padded.1.shape)
                        records.append(["label": fixture.label, "round": round,
                            "kind": "router-pad-to-513-prefix-width", "boundary": boundary,
                            "stage": full.0,
                            "exact": arrayEqual(fullPrefix, padded.1).item(Bool.self),
                            "max_error": abs(fullPrefix.asType(.float32)
                                - padded.1.asType(.float32)).max().item(Float.self)])
                    }
                    try write()
                }
                if round == 0, fixture.label != "0.5K",
                   env["AFM_TEST_ONE_PASS_STAGE_OTHER"] == "1" {
                    let boundary = ids.count - 31
                    let splitTarget = model.newCache(parameters: nil)
                    let onePassTarget = model.newCache(parameters: nil)
                    let splitHead = head.newCache()
                    let onePassHead = head.newCache()
                    let splitBridge = snapshots["split"]!.restoreForTesting(
                        target: splitTarget, head: splitHead)
                    let onePassBridge = snapshots["one-pass"]!.restoreForTesting(
                        target: onePassTarget, head: onePassHead)
                    var cacheDifferences = [[String: Any]]()
                    if !arrayEqual(splitBridge.hidden, onePassBridge.hidden).item(Bool.self) {
                        cacheDifferences.append(["state": "hidden"])
                    }
                    if !arrayEqual(splitBridge.stream, onePassBridge.stream).item(Bool.self) {
                        cacheDifferences.append(["state": "stream"])
                    }
                    for (layer, pair) in zip(splitTarget + splitHead,
                                              onePassTarget + onePassHead).enumerated() {
                        if pair.0.offset != pair.1.offset {
                            cacheDifferences.append(["layer": layer, "state": "offset"])
                        }
                        let left = cacheArrays(pair.0), right = cacheArrays(pair.1)
                        for slot in left.indices {
                            switch (left[slot], right[slot]) {
                            case (.none, .none): break
                            case (.some(let x), .some(let y)):
                                if x.shape != y.shape || x.dtype != y.dtype
                                    || !arrayEqual(x, y).item(Bool.self) {
                                    cacheDifferences.append(["layer": layer, "slot": slot,
                                        "shape_split": x.shape, "shape_one_pass": y.shape])
                                }
                            default:
                                cacheDifferences.append(["layer": layer, "slot": slot,
                                    "state": "presence"])
                            }
                        }
                    }
                    records.append(["label": fixture.label, "round": round,
                        "kind": "native-other-snapshot-difference",
                        "differences": cacheDifferences])
                    let fullIDs = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
                    let prefixIDs = MLXArray(Array(ids.prefix(boundary)).map(Int32.init))
                        .reshaped(1, -1)
                    let fullStreams = model.layerStreamsForTesting(
                        inputIDs: fullIDs, cache: model.newCache(parameters: nil))
                    let prefixStreams = model.layerStreamsForTesting(
                        inputIDs: prefixIDs, cache: model.newCache(parameters: nil))
                    for index in 0..<min(fullStreams.count, prefixStreams.count) {
                        let fullPrefix = fullStreams[index][0..., ..<boundary]
                        let prefix = prefixStreams[index]
                        XCTAssertEqual(fullPrefix.shape, prefix.shape)
                        let exact = arrayEqual(fullPrefix, prefix).item(Bool.self)
                        records.append(["label": fixture.label, "round": round,
                            "kind": "native-other-layer-prefix-width", "boundary": boundary,
                            "stage": index == 0 ? "embedding"
                                : (index == fullStreams.count - 1 ? "final_mixer"
                                    : "decoder_\(index - 1)"),
                            "exact": exact,
                            "max_error": abs(fullPrefix.asType(.float32)
                                - prefix.asType(.float32)).max().item(Float.self)])
                        if !exact { break }
                    }
                    if fixture.label == "2K",
                       env["AFM_TEST_ONE_PASS_SUFFIX_STAGE"] == "1" {
                        let suffixCache = model.newCache(parameters: nil)
                        let prefixForward = model.layerStreamsForTesting(
                            inputIDs: prefixIDs, cache: suffixCache)
                        eval(prefixForward.last!)
                        let suffixIDs = MLXArray(Array(ids.suffix(31)).map(Int32.init))
                            .reshaped(1, -1)
                        let suffixStreams = model.layerStreamsForTesting(
                            inputIDs: suffixIDs, cache: suffixCache)
                        for index in 0..<min(fullStreams.count, suffixStreams.count) {
                            let fullSuffix = fullStreams[index][0..., boundary...]
                            let suffix = suffixStreams[index]
                            XCTAssertEqual(fullSuffix.shape, suffix.shape)
                            let exact = arrayEqual(fullSuffix, suffix).item(Bool.self)
                            records.append(["label": fixture.label, "round": round,
                                "kind": "native-full-versus-recomputed-suffix",
                                "stage": index == 0 ? "embedding"
                                    : (index == fullStreams.count - 1 ? "final_mixer"
                                        : "decoder_\(index - 1)"),
                                "exact": exact,
                                "max_error": abs(fullSuffix.asType(.float32)
                                    - suffix.asType(.float32)).max().item(Float.self)])
                            if !exact { break }
                        }
                        let fullSuffixStages = model.firstLinearLayerStageTraceForTesting(
                            inputIDs: fullIDs, cache: model.newCache(parameters: nil))
                        let stageCache = model.newCache(parameters: nil)
                        let stagePrefix = model.layerStreamsForTesting(
                            inputIDs: prefixIDs, cache: stageCache)
                        eval(stagePrefix.last!)
                        let recomputedStages = model.firstLinearLayerStageTraceForTesting(
                            inputIDs: suffixIDs, cache: stageCache)
                        XCTAssertEqual(fullSuffixStages.map(\.0), recomputedStages.map(\.0))
                        let hcMixedIndex = try XCTUnwrap(fullSuffixStages.firstIndex {
                            $0.0 == "hc_mixed"
                        })
                        let hcReadIndex = try XCTUnwrap(fullSuffixStages.firstIndex {
                            $0.0 == "hc_read"
                        })
                        XCTAssertTrue(arrayEqual(fullSuffixStages[hcMixedIndex].1,
                            fullSuffixStages[hcReadIndex].1).item(Bool.self))
                        XCTAssertTrue(arrayEqual(recomputedStages[hcMixedIndex].1,
                            recomputedStages[hcReadIndex].1).item(Bool.self))
                        XCTAssertTrue(arrayEqual(fullSuffixStages.last!.1,
                            fullStreams[1]).item(Bool.self))
                        XCTAssertTrue(arrayEqual(recomputedStages.last!.1,
                            suffixStreams[1]).item(Bool.self))
                        for (full, recomputed) in zip(fullSuffixStages, recomputedStages) {
                            let fullSuffix = full.1[0..., boundary...]
                            XCTAssertEqual(fullSuffix.shape, recomputed.1.shape)
                            records.append(["label": fixture.label, "round": round,
                                "kind": "native-decoder0-full-versus-suffix-stage",
                                "stage": full.0,
                                "exact": arrayEqual(fullSuffix, recomputed.1).item(Bool.self),
                                "max_error": abs(fullSuffix.asType(.float32)
                                    - recomputed.1.asType(.float32)).max().item(Float.self)])
                        }
                        for hcPadTo in (env["AFM_TEST_ONE_PASS_HC_PAD_TO"] ?? "")
                            .split(separator: ",").compactMap({ Int($0) }) {
                            let paddedCache = model.newCache(parameters: nil)
                            let paddedPrefix = model.layerStreamsForTesting(
                                inputIDs: prefixIDs, cache: paddedCache)
                            eval(paddedPrefix.last!)
                            let paddedStages = model.firstLinearLayerStageTraceForTesting(
                                inputIDs: suffixIDs, cache: paddedCache,
                                hcDownPadTo: hcPadTo,
                                hcDownForceUnsplit:
                                    env["AFM_TEST_ONE_PASS_HC_PAD_UNSPLIT"] == "1")
                            XCTAssertEqual(fullSuffixStages.map(\.0), paddedStages.map(\.0))
                            for (full, padded) in zip(fullSuffixStages, paddedStages) {
                                let fullSuffix = full.1[0..., boundary...]
                                XCTAssertEqual(fullSuffix.shape, padded.1.shape)
                                records.append(["label": fixture.label, "round": round,
                                    "kind": "native-hc-down-padded-full-versus-suffix",
                                    "pad_to": hcPadTo,
                                    "force_unsplit": env["AFM_TEST_ONE_PASS_HC_PAD_UNSPLIT"] == "1",
                                    "stage": full.0,
                                    "exact": arrayEqual(fullSuffix, padded.1).item(Bool.self),
                                    "max_error": abs(fullSuffix.asType(.float32)
                                        - padded.1.asType(.float32)).max().item(Float.self)])
                            }
                        }
                        if env["AFM_TEST_ONE_PASS_HC_UNSPLIT"] == "1" {
                            let unsplitCache = model.newCache(parameters: nil)
                            let unsplitPrefix = model.layerStreamsForTesting(
                                inputIDs: prefixIDs, cache: unsplitCache)
                            eval(unsplitPrefix.last!)
                            let unsplitStages = model.firstLinearLayerStageTraceForTesting(
                                inputIDs: suffixIDs, cache: unsplitCache,
                                hcDownForceUnsplit: true)
                            XCTAssertEqual(fullSuffixStages.map(\.0), unsplitStages.map(\.0))
                            for (full, unsplit) in zip(fullSuffixStages, unsplitStages) {
                                let fullSuffix = full.1[0..., boundary...]
                                XCTAssertEqual(fullSuffix.shape, unsplit.1.shape)
                                records.append(["label": fixture.label, "round": round,
                                    "kind": "native-hc-down-unsplit-full-versus-suffix",
                                    "stage": full.0,
                                    "exact": arrayEqual(fullSuffix, unsplit.1).item(Bool.self),
                                    "max_error": abs(fullSuffix.asType(.float32)
                                        - unsplit.1.asType(.float32)).max().item(Float.self)])
                            }
                        }
                    }
                    let fullStages = model.firstLinearLayerStageTraceForTesting(
                        inputIDs: fullIDs, cache: model.newCache(parameters: nil))
                    let prefixStages = model.firstLinearLayerStageTraceForTesting(
                        inputIDs: prefixIDs, cache: model.newCache(parameters: nil))
                    XCTAssertTrue(arrayEqual(fullStages.last!.1, fullStreams[1]).item(Bool.self))
                    XCTAssertTrue(arrayEqual(prefixStages.last!.1, prefixStreams[1]).item(Bool.self))
                    for (full, short) in zip(fullStages, prefixStages) {
                        let fullPrefix = full.1[0..., ..<boundary]
                        XCTAssertEqual(fullPrefix.shape, short.1.shape)
                        records.append(["label": fixture.label, "round": round,
                            "kind": "native-other-decoder0-stage-prefix-width",
                            "boundary": boundary, "stage": full.0,
                            "exact": arrayEqual(fullPrefix, short.1).item(Bool.self),
                            "max_error": abs(fullPrefix.asType(.float32)
                                - short.1.asType(.float32)).max().item(Float.self)])
                    }
                    try write()
                }
                // Outside timings: verify all live cache slots, including the
                // sixth derived score bank. Both full-forward paths must match.
                let a = sessions["endpoint"]!.cacheArraysForTesting()
                let b = sessions["one-pass"]!.cacheArraysForTesting()
                XCTAssertEqual(a.count, b.count)
                for (index, pair) in zip(a, b).enumerated() {
                    XCTAssertEqual(pair.0 == nil, pair.1 == nil, "nil slot \(index)")
                    if let x = pair.0, let y = pair.1 {
                        XCTAssertEqual(x.shape, y.shape)
                        XCTAssertTrue(arrayEqual(x, y).item(Bool.self), "cold cache \(index)")
                    }
                }
                var output = [String: [Int]]()
                for i in 0..<3 {
                    let arm = order[(i + round) % order.count]
                    let result = drain(sessions[arm]!)
                    output[arm] = result.tokens
                    let snapshot = snapshots[arm]!
                    records.append(["label": fixture.label, "round": round, "warmup": round == 0,
                        "arm": arm, "kind": "cold", "prompt_tokens": ids.count,
                        "prompt_sha256": hash(ids), "tokens": result.tokens,
                        "text": context.tokenizer.decode(tokens: result.tokens),
                        "prefill_ms": timing[arm]!, "decode_ms": result.milliseconds,
                        "snapshot_tokens": snapshot.promptIds.count,
                        "snapshot_bytes": snapshot.estimatedRetainedBytes,
                        "verification_cycles": sessions[arm]!.verificationCycleCount])
                }
                XCTAssertEqual(output["endpoint"], output["one-pass"], "native cold output preservation")
                for arm in ["split", "one-pass"] {
                    let generator = arm == "one-pass" ? candidate : control
                    let start = DispatchTime.now().uptimeNanoseconds
                    let replay = try XCTUnwrap(generator.makeSession(promptIds: ids, maxTokens: 128,
                        promptState: snapshots[arm], allowPromptPrefixReplay: true, prefillStepSize: step))
                    let prefill = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    let result = drain(replay)
                    records.append(["label": fixture.label, "round": round, "warmup": round == 0,
                        "arm": arm, "kind": "repeat", "prompt_tokens": ids.count,
                        "prompt_sha256": hash(ids), "tokens": result.tokens,
                        "text": context.tokenizer.decode(tokens: result.tokens),
                        "prefill_ms": prefill, "decode_ms": result.milliseconds,
                        "cold_exact": result.tokens == output[arm],
                        "endpoint_exact": result.tokens == output["endpoint"],
                        "verification_cycles": replay.verificationCycleCount])
                }
                try write()
                print("ONE_PASS_NATIVE label=\(fixture.label) round=\(round) records=\(records.count)")
            }
        }
        #endif
    }
}
