import Foundation
import Jinja
import XCTest
@testable import AFMKitMLX

final class MLXHistoryTemplatePolicyTests: XCTestCase {
    func testNativeTemplateSelectionAndMissingModelDoNotInheritCapability() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertNil(MLXHistoryTemplatePolicy.nativeTemplate(directory: directory))
        try "preserve_thinking".write(to: directory.appendingPathComponent("chat_template.jinja"),
            atomically: true, encoding: .utf8)
        XCTAssertEqual(MLXHistoryTemplatePolicy.nativeTemplate(directory: directory), "preserve_thinking")
        let config = try JSONSerialization.data(withJSONObject: ["chat_template": "{{ messages }}"])
        try config.write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let effective = MLXHistoryTemplatePolicy.nativeTemplate(directory: directory)
        XCTAssertEqual(effective, "{{ messages }}")
        XCTAssertNil(MLXHistoryTemplatePolicy.applyingDefaults([:], template: effective)["preserve_thinking"])
        XCTAssertNil(MLXHistoryTemplatePolicy.nativeTemplate(directory: directory.appendingPathComponent("missing")))
    }

    func testSupportedTemplateDefaultsToDroppingPriorThinking() {
        let kwargs = MLXHistoryTemplatePolicy.applyingDefaults(["enable_thinking": true],
            template: QwenNextToolTemplateFixture.template)
        XCTAssertEqual(kwargs["preserve_thinking"] as? Bool, false)
        XCTAssertEqual(kwargs["enable_thinking"] as? Bool, true)
    }

    func testExplicitOverrideAndUnsupportedTemplateRemainUnchanged() {
        for override in [true, false] {
            let kwargs = MLXHistoryTemplatePolicy.applyingDefaults(["preserve_thinking": override],
                template: QwenNextToolTemplateFixture.template)
            XCTAssertEqual(kwargs["preserve_thinking"] as? Bool, override)
        }
        for template in [nil, "{{ messages }}"] as [String?] {
            XCTAssertNil(MLXHistoryTemplatePolicy.applyingDefaults([:], template: template)["preserve_thinking"])
        }
    }

    func testNativeTemplateDropsPreviousTurnButKeepsCurrentTurnReasoning() throws {
        let messages: [[String: Any]] = [
            ["role": "user", "content": "first"],
            ["role": "assistant", "content": "answer", "reasoning_content": "OLD_REASON"],
            ["role": "user", "content": "followup"],
            ["role": "assistant", "content": "current", "reasoning_content": "NEW_REASON"]
        ]
        let template = try Template(QwenNextToolTemplateFixture.template)
        func render(_ input: [String: any Sendable]) throws -> String {
            let kwargs = MLXHistoryTemplatePolicy.applyingDefaults(input,
                template: QwenNextToolTemplateFixture.template)
            var context: [String: Value] = ["messages": try Value(any: messages),
                "tools": try Value(any: [] as [[String: Any]]), "add_generation_prompt": try Value(any: true)]
            for (key, value) in kwargs { context[key] = try Value(any: value) }
            return try template.render(context)
        }
        let normal = try render(["enable_thinking": true])
        XCTAssertFalse(normal.contains("OLD_REASON"))
        XCTAssertTrue(normal.contains("NEW_REASON"))
        let preserved = try render(["enable_thinking": true, "preserve_thinking": true])
        XCTAssertTrue(preserved.contains("OLD_REASON"))
        XCTAssertTrue(preserved.contains("NEW_REASON"))
    }
}
