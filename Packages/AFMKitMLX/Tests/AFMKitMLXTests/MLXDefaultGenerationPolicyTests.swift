import XCTest
@testable import AFMKitMLX

final class MLXDefaultGenerationPolicyTests: XCTestCase {
    func testDefaultMaximumResponseTokensMatchesAFMRuntimePolicy() {
        XCTAssertEqual(MLXModelService.defaultMaximumResponseTokens, 8_192)
    }
}
