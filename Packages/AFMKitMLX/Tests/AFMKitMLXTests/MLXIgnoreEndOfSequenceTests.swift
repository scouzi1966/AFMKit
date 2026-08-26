import XCTest
@preconcurrency import MLXLMCommon

@testable import AFMKitMLX

final class MLXIgnoreEndOfSequenceTests: XCTestCase {
    func testGenerateParametersDefaultToStoppingAtEOS() {
        XCTAssertFalse(GenerateParameters().ignoreEndOfSequence)
    }

    func testGenerateParametersCanIgnoreEOS() {
        XCTAssertTrue(GenerateParameters(ignoreEndOfSequence: true).ignoreEndOfSequence)
    }

    func testEOSStopsByDefault() {
        XCTAssertEqual(
            BatchScheduler.tokenDisposition(
                tokenCount: 0,
                maxTokens: 10,
                tokenID: 2,
                unknownTokenID: 0,
                ignoreEndOfSequence: false,
                eosTokenIDs: [2],
                consecutiveSuppressedEndOfSequenceTokens: 0
            ),
            .stop
        )
    }

    func testIgnoredEOSIsSuppressedWithoutConsumingVisibleTokenBudget() {
        XCTAssertEqual(
            BatchScheduler.tokenDisposition(
                tokenCount: 3,
                maxTokens: 4,
                tokenID: 2,
                unknownTokenID: 0,
                ignoreEndOfSequence: true,
                eosTokenIDs: [2],
                consecutiveSuppressedEndOfSequenceTokens: 0
            ),
            .suppress
        )
    }

    func testIgnoredEOSHasConsecutiveSafetyLimit() {
        XCTAssertEqual(
            BatchScheduler.tokenDisposition(
                tokenCount: 0,
                maxTokens: 10,
                tokenID: 2,
                unknownTokenID: 0,
                ignoreEndOfSequence: true,
                eosTokenIDs: [2],
                consecutiveSuppressedEndOfSequenceTokens:
                    BatchScheduler.maximumConsecutiveSuppressedEndOfSequenceTokens - 1
            ),
            .stop
        )
    }

    func testVisibleTokenBudgetWinsBeforeEOSPolicy() {
        XCTAssertEqual(
            BatchScheduler.tokenDisposition(
                tokenCount: 4,
                maxTokens: 4,
                tokenID: 9,
                unknownTokenID: 0,
                ignoreEndOfSequence: true,
                eosTokenIDs: [2],
                consecutiveSuppressedEndOfSequenceTokens: 0
            ),
            .length
        )
    }
}
