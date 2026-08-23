import Foundation
import XCTest
@testable import AFMEvalKit

final class AFMEvalKitTests: XCTestCase {
    func testSchemaRoundTripsAndMergesParameters() throws {
        let suite = AFMEvaluationSuite(
            name: "smoke",
            description: "Provider-free evaluation schema.",
            defaults: .init(temperature: 0, maxTokens: 64),
            cases: [.init(id: "hello", prompt: "Say hello")])
        let decoded = try JSONDecoder().decode(
            AFMEvaluationSuite.self,
            from: JSONEncoder().encode(suite))
        XCTAssertEqual(decoded.name, "smoke")
        XCTAssertEqual(
            decoded.defaults?.merging(.init(maxTokens: 128)).maxTokens,
            128)
        XCTAssertEqual(
            decoded.defaults?.merging(.init(maxTokens: 128)).temperature,
            0)
    }

    func testValidatorRejectsUnknownKeysAndUnsafeBounds() throws {
        let unknown = Data("""
        {"schemaVersion":1,"name":"bad","description":"Bad.","surprise":true,
         "cases":[{"id":"x","prompt":"hello"}]}
        """.utf8)
        XCTAssertThrowsError(try AFMEvaluationValidator.decode(unknown)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Unknown key"))
        }

        let invalid = AFMEvaluationSuite(
            name: "bad",
            description: "Bad bounds.",
            cases: [.init(
                id: "x",
                prompt: "hello",
                expectations: .init(minimumCharacters: 10, maximumCharacters: 5))])
        XCTAssertThrowsError(try AFMEvaluationValidator.validate(invalid))
    }

    func testCrossCaseMatchingRequiresExplicitTrustAndEarlierCase() throws {
        let suite = AFMEvaluationSuite(
            name: "matching",
            description: "Trusted matching.",
            cases: [
                .init(id: "first", prompt: "one"),
                .init(
                    id: "second",
                    prompt: "two",
                    expectations: .init(matchesCase: "first"))
            ])
        XCTAssertThrowsError(try AFMEvaluationValidator.validate(suite))
        XCTAssertNoThrow(try AFMEvaluationValidator.validate(
            suite,
            allowsCrossCaseMatching: true))
    }

    func testDeterministicScoringHasNoJudgeDependency() {
        let scored = AFMEvaluationScorer.score(
            output: "{\"answer\":\"Hello\"}",
            toolCallNames: ["lookup"],
            expectations: .init(
                contains: ["hello"],
                notContains: ["secret"],
                validJSON: true,
                toolCallName: "lookup"))
        XCTAssertEqual(scored.0, .passed)
        XCTAssertTrue(scored.1.allSatisfy(\.passed))
        XCTAssertEqual(
            AFMEvaluationScorer.score(output: "anything", expectations: nil).0,
            .observed)
    }

    func testMetricsAndBudgetAreDeterministic() throws {
        XCTAssertEqual(
            AFMEvaluationRunPolicy.tokensPerSecond(
                completionTokens: 50,
                generationTime: 2,
                duration: 3),
            25)
        let oversized = AFMEvaluationSuite(
            name: "oversized",
            description: "Exceeds budget.",
            defaults: .init(maxTokens: 32_768),
            cases: (0..<31).map { .init(id: "case-\($0)", prompt: "test") })
        XCTAssertThrowsError(try AFMEvaluationRunPolicy.validatePlannedOutput(
            suites: [oversized],
            baseParameters: .init(maxTokens: 256)))

        let invalid = AFMEvaluationSuite(
            name: "invalid",
            description: "Programmatic suite still requires validation.",
            cases: [.init(id: "duplicate", prompt: "one"),
                    .init(id: "duplicate", prompt: "two")])
        XCTAssertThrowsError(try AFMEvaluationRunPolicy.validatePlannedOutput(
            suites: [invalid],
            baseParameters: .init(maxTokens: 16)))
    }

    func testValidatorRejectsUnsafeNestedGenerationConfiguration() throws {
        XCTAssertThrowsError(try AFMEvaluationValidator.validateParameters(
            .init(topLogprobs: 3),
            context: "case"))
        XCTAssertThrowsError(try AFMEvaluationValidator.validateParameters(
            .init(stop: [""]),
            context: "case"))
        XCTAssertThrowsError(try AFMEvaluationValidator.validateParameters(
            .init(responseFormat: .init(type: "json_schema")),
            context: "case"))
    }

    func testReportWriterEscapesUntrustedText() {
        let parameters = AFMEvaluationParameters(maxTokens: 16)
        let result = AFMEvaluationCaseResult(
            suite: "suite<script>", caseID: "case", prompt: "<img>", system: nil,
            output: "<script>alert('x')</script>", reasoning: "<reasoning>",
            toolCalls: [.init(name: "lookup", arguments: "</pre><script>x</script>")],
            outcome: .error,
            checks: [.init(name: "<check>", passed: false, detail: "<detail>")],
            error: "bad & worse",
            startedAt: Date(timeIntervalSince1970: 0), durationSeconds: 1,
            timeToFirstTokenSeconds: nil, promptTimeSeconds: nil,
            generationTimeSeconds: nil, promptTokens: 1, cachedPromptTokens: 0,
            completionTokens: 1, tokensPerSecond: 1, finishReason: "error",
            parameters: parameters)
        let report = AFMEvaluationRunReport(
            afmVersion: "test", model: "model<script>", suites: ["suite"],
            startedAt: Date(timeIntervalSince1970: 0),
            finishedAt: Date(timeIntervalSince1970: 1), interrupted: false,
            reproducibilityCommand: "afm --eval", system: .init(
                operatingSystem: "macOS", architecture: "arm64",
                processorCount: 8, physicalMemoryBytes: 16_000_000_000),
            results: [result])
        let html = AFMEvaluationReportWriter.html(for: report)
        XCTAssertFalse(html.contains("<script>alert('x')</script>"))
        XCTAssertFalse(html.contains("</pre><script>x</script>"))
        XCTAssertTrue(html.contains("bad &amp; worse"))
    }
}
