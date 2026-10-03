import Foundation
import CryptoKit
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Direct-model lifecycle and known-answer screen, not an HTTP grammar or
/// scheduler-concurrency qualification. Explicit local weights/report required.
final class QwenOnePassPromptGrowingTests: XCTestCase {
    func testOptionalNativeGrowingStructuredTasks() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_ONE_PASS_MODEL"],
              let reportPath = env["AFM_TEST_ONE_PASS_GROWING_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint and growing-task report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only qualification")
        #else
        continueAfterFailure = false
        let reportURL = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard reportURL.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: reportURL.path) else {
            throw NSError(domain: "OnePassGrowing", code: 1)
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let directory = URL(fileURLWithPath: modelPath)
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: directory))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let head = try model.loadEmbeddedMTPHead(modelDirectory: directory, groupSize: 32, bits: 4)
        let control = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .strictSingletonEquivalent, draftDispatchStride: 0)
        let candidate = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .strictSingletonEquivalent, draftDispatchStride: 0, onePassPromptCapture: true)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        let maxTokens = 192
        let backoff = 31
        let contextRowsToTest = try (env["AFM_TEST_ONE_PASS_ROWS"] ?? "16,80").split(separator: ",").map {
            guard let value = Int($0), (1...224).contains(value) else {
                throw NSError(domain: "OnePassGrowing", code: 2)
            }
            return value
        }
        XCTAssertFalse(contextRowsToTest.isEmpty)
        XCTAssertEqual(Set(contextRowsToTest).count, contextRowsToTest.count)
        let eos = context.configuration.resolvedEOSTokenIds(tokenizer: context.tokenizer)
        struct TaskCase {
            let name: String
            let question: String
            let expected: String
        }
        let cases = [
            TaskCase(name: "inventory", question: "Agent inventory task: alpha has 7 units, beta has 11, gamma has 3. Return only JSON with key total and their integer sum.", expected: "{\"total\":21}"),
            TaskCase(name: "filter", question: "Agent triage task: incidents A severity 2, B severity 5, C severity 4, D severity 1. Escalate severity >=4, sorted by ID. Return only JSON with key escalate and the array of IDs.", expected: "{\"escalate\":[\"B\",\"C\"]}"),
            TaskCase(name: "tool-plan", question: "Agent tool decision: available actions are read_file and write_file. You must inspect src/cache.swift before editing it. Select the next action, not the later edit. Return only JSON with keys action and path.", expected: "{\"action\":\"read_file\",\"path\":\"src/cache.swift\"}"),
            TaskCase(name: "policy", question: "Agent deployment task: deploy only if tests_pass AND review_approved. tests_pass=true, review_approved=false. Return only JSON with keys deploy (boolean) and blocker (the name of the false condition).", expected: "{\"deploy\":false,\"blocker\":\"review_approved\"}"),
        ]
        var records = [[String: Any]]()
        var pairedRegressions = [String]()
        var cancellations = [[String: Any]]()
        func write() throws {
            try JSONSerialization.data(withJSONObject: [
                "scope": "Direct native model, greedy MTP depth3. Prompt-only JSON; no grammar processor, HTTP, actual tools or multi-request GPU batch. Fixed-order times are diagnostic, not benchmark results.",
                "model": modelPath, "prefill_step": step, "max_tokens": maxTokens,
                "context_rows": contextRowsToTest,
                "backoff": backoff, "temperature": 0, "top_p": 1, "depth": 3,
                "eos_ids": eos.sorted(), "cancellations": cancellations,
                "paired_quality_regressions": pairedRegressions, "records": records,
            ], options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
        }
        func ids(_ messages: [[String: String]]) throws -> [Int] {
            try context.tokenizer.applyChatTemplate(messages: messages, tools: nil,
                additionalContext: ["enable_thinking": false])
        }
        func hash(_ tokens: [Int]) -> String {
            let data = tokens.map(Int32.init).withUnsafeBufferPointer { Data(buffer: $0) }
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        func session(_ prompt: [Int], onePass: Bool, snapshot: Qwen4ExpMTPPromptState? = nil,
                     retain: Bool = false, offset: Int = 0) throws -> (Qwen4ExpMTPSession, Double) {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try XCTUnwrap((onePass ? candidate : control).makeSession(
                promptIds: prompt, maxTokens: maxTokens, eosIds: eos,
                promptState: snapshot, retainPromptState: retain,
                allowPromptPrefixReplay: snapshot != nil, prefillStepSize: step,
                promptSnapshotBackoffTokens: offset))
            return (result, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        func drain(_ session: Qwen4ExpMTPSession) -> [Int] {
            var result = [Int]()
            while let token = session.nextToken() { result.append(token) }
            return result
        }
        func visibleText(_ tokens: [Int]) -> String {
            // Session yields its EOS to its caller. Match the serving boundary:
            // omit only recognized terminal EOS IDs, preserving all raw tokens.
            let visible = tokens.last.map(eos.contains) == true ? Array(tokens.dropLast()) : tokens
            return context.tokenizer.decode(tokens: visible).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        func valid(_ tokens: [Int], expected: String) -> Bool {
            guard let value = try? JSONSerialization.jsonObject(with: Data(visibleText(tokens).utf8)),
                  let wanted = try? JSONSerialization.jsonObject(with: Data(expected.utf8)),
                  let a = try? JSONSerialization.data(withJSONObject: value, options: .sortedKeys),
                  let b = try? JSONSerialization.data(withJSONObject: wanted, options: .sortedKeys) else { return false }
            // Canonical JSON distinguishes false from numeric 0, unlike NSNumber equality.
            return a == b
        }
        func record(_ label: String, _ arm: String, _ prompt: [Int], _ tokens: [Int],
                    _ expected: String, _ ms: Double, reused: Int = 0) {
            records.append(["label": label, "arm": arm, "prompt_tokens": prompt.count,
                "prompt_sha256": hash(prompt), "tokens": tokens,
                "text": visibleText(tokens), "raw_text": context.tokenizer.decode(tokens: tokens), "expected_json": expected,
                "known_answer_pass": valid(tokens, expected: expected),
                "constructor_ms": ms, "reused_prefix_tokens": reused])
        }
        // Explicit row counts select prompt lengths. Record actual tokens;
        // row count is NOT a token count or a claim of sparse-QSA coverage.
        for contextRows in contextRowsToTest {
            let background = (0..<contextRows).map {
                "Repository note \($0): changes require tests and review; archive logs are context only."
            }.joined(separator: "\n")
            let seedMessages = [["role": "system", "content": "You are a coding assistant. Follow the latest user task exactly. Return only the requested JSON object, without markdown or explanation."],
                ["role": "user", "content": background + "\nAcknowledge with READY."]]
            let seedIDs = try ids(seedMessages)
            let frontier = seedIDs.count - backoff
            XCTAssertGreaterThan(frontier, 0)
            // Two immutable snapshots with the same ownership, but different
            // cold prefill geometry. Decode the donor AFTER taking each snapshot.
            let (oneSeed, _) = try session(seedIDs, onePass: true, retain: true, offset: backoff)
            let oneSnapshot = try XCTUnwrap(oneSeed.takePromptState())
            let (splitSeed, _) = try session(seedIDs, onePass: false, retain: true, offset: backoff)
            let splitSnapshot = try XCTUnwrap(splitSeed.takePromptState())
            XCTAssertEqual(oneSnapshot.promptIds, Array(seedIDs.prefix(frontier)))
            XCTAssertEqual(splitSnapshot.promptIds, oneSnapshot.promptIds)
            _ = drain(oneSeed)
            _ = drain(splitSeed)
            for task in cases {
                let label = "rows\(contextRows)-\(task.name)"
                let messages = seedMessages + [["role": "assistant", "content": "READY"],
                    ["role": "user", "content": task.question]]
                let prompt = try ids(messages)
                XCTAssertTrue(prompt.starts(with: oneSnapshot.promptIds))
                let (cold, coldMs) = try session(prompt, onePass: false)
                let coldOutput = drain(cold)
                record(label, "endpoint-cold", prompt, coldOutput, task.expected, coldMs)
                let (captured, capturedMs) = try session(prompt, onePass: true, retain: true, offset: backoff)
                let capturedOutput = drain(captured)
                XCTAssertEqual(coldOutput, capturedOutput, "one-pass cold preservation \(label)")
                record(label, "one-pass-cold", prompt, capturedOutput, task.expected, capturedMs)
                let (split, splitMs) = try session(prompt, onePass: false, snapshot: splitSnapshot,
                    retain: true, offset: backoff)
                let splitNext = try XCTUnwrap(split.takePromptState())
                let splitOutput = drain(split)
                record(label, "split-growing", prompt, splitOutput, task.expected, splitMs, reused: frontier)
                let (grown, grownMs) = try session(prompt, onePass: true, snapshot: oneSnapshot,
                    retain: true, offset: backoff)
                let nextSnapshot = try XCTUnwrap(grown.takePromptState())
                XCTAssertEqual(nextSnapshot.promptIds, Array(prompt.dropLast(backoff)))
                XCTAssertEqual(nextSnapshot.promptIds, splitNext.promptIds)
                // Interleave another restored owner, cancel it mid-generation,
                // then continue this owner. This tests state isolation, NOT batching.
                let alternatePrompt = try ids(seedMessages + [["role": "assistant", "content": "READY"],
                    ["role": "user", "content": "Return only JSON with key steps containing an array of every integer from 1 through 100, in order."]])
                XCTAssertNotEqual(alternatePrompt, prompt)
                XCTAssertTrue(alternatePrompt.starts(with: oneSnapshot.promptIds))
                let (cancelled, _) = try session(alternatePrompt, onePass: true, snapshot: oneSnapshot)
                var grownOutput = [Int]()
                var cancelledTokens = [Int]()
                for _ in 0..<7 {
                    if let token = grown.nextToken() { grownOutput.append(token) }
                    let token = try XCTUnwrap(cancelled.nextToken())
                    XCTAssertFalse(eos.contains(token), "Cancellation fixture ended too early")
                    cancelledTokens.append(token)
                }
                let stillActive = try XCTUnwrap(cancelled.nextToken())
                XCTAssertFalse(eos.contains(stillActive))
                cancelledTokens.append(stillActive)
                XCTAssertLessThan(cancelledTokens.count, maxTokens)
                cancelled.cancel()
                XCTAssertNil(cancelled.nextToken())
                cancellations.append(["label": label, "prompt_sha256": hash(alternatePrompt),
                    "tokens_before_cancel": cancelledTokens, "non_eos_before_cancel": true,
                    "nil_after_cancel": true])
                grownOutput += drain(grown)
                record(label, "one-pass-growing-interleaved", prompt, grownOutput, task.expected, grownMs, reused: frontier)
                let (repeatGrow, repeatMs) = try session(prompt, onePass: true, snapshot: oneSnapshot,
                    retain: true, offset: backoff)
                let repeatOutput = drain(repeatGrow)
                XCTAssertEqual(grownOutput, repeatOutput, "interleaved/cancelled isolation \(label)")
                record(label, "one-pass-growing-repeat", prompt, repeatOutput, task.expected, repeatMs, reused: frontier)
                if valid(splitOutput, expected: task.expected) && !valid(grownOutput, expected: task.expected) {
                    pairedRegressions.append(label + ":split-growing->one-pass-growing")
                }
                if valid(coldOutput, expected: task.expected) && !valid(grownOutput, expected: task.expected) {
                    pairedRegressions.append(label + ":endpoint-cold->one-pass-growing")
                }
                // Restore the newly captured frontier for a second conversation
                // growth. Fixed assistant text makes all arms use identical IDs.
                let nextMessages = messages + [["role": "assistant", "content": task.expected],
                    ["role": "user", "content": "New independent check: return only JSON with key sum and integer value equal to 17 plus 25."]]
                let nextPrompt = try ids(nextMessages)
                let nextExpected = "{\"sum\":42}"
                XCTAssertTrue(nextPrompt.starts(with: nextSnapshot.promptIds))
                var nextOutputs = [String: [Int]]()
                for arm in ["endpoint-second", "split-second", "one-pass-second"] {
                    let state: Qwen4ExpMTPPromptState? = arm == "endpoint-second" ? nil
                        : (arm == "split-second" ? splitNext : nextSnapshot)
                    let (request, ms) = try session(nextPrompt, onePass: arm == "one-pass-second", snapshot: state)
                    let output = drain(request)
                    nextOutputs[arm] = output
                    record(label, arm, nextPrompt, output, nextExpected, ms,
                        reused: state?.promptIds.count ?? 0)
                }
                for baseline in ["endpoint-second", "split-second"] {
                    if valid(nextOutputs[baseline]!, expected: nextExpected)
                        && !valid(nextOutputs["one-pass-second"]!, expected: nextExpected) {
                        pairedRegressions.append(label + ":" + baseline + "->one-pass-second")
                    }
                }
                try write()
                print("ONE_PASS_GROWING label=\(label) records=\(records.count)")
            }
        }
        XCTAssertEqual(records.count, cases.count * contextRowsToTest.count * 8)
        // Save every task, including failed model answers, before a quality gate.
        XCTAssertTrue(pairedRegressions.isEmpty, "Candidate quality regressions: \(pairedRegressions)")
        XCTAssertTrue(records.allSatisfy { $0["known_answer_pass"] as? Bool == true },
            "Known-answer floor failed; inspect raw report, do not attribute automatically to model quality")
        #endif
    }
}
