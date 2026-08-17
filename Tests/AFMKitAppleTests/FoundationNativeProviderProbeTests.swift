import Foundation
import FoundationModels
import XCTest
@testable import AFMKitApple

@available(macOS 27.0, *)
final class FoundationNativeProviderProbeTests: XCTestCase {
    func testMissingEntitlementSnapshotDoesNotProbePCC() {
        let probe = AFMFoundationNativeProviderProbe()
        let snapshot = probe.privateCloudComputeSnapshot(
            hasEntitlement: false,
            locale: Locale(identifier: "en_US")
        )

        XCTAssertFalse(snapshot.hasEntitlement)
        XCTAssertEqual(snapshot.entitlement, "com.apple.developer.private-cloud-compute")
        XCTAssertEqual(snapshot.localeIdentifier, "en_US")
        XCTAssertFalse(snapshot.localeSupported)
        XCTAssertEqual(snapshot.quotaStatus, "unknown")
        XCTAssertFalse(snapshot.quotaIsLimitReached)
        XCTAssertNil(snapshot.quotaLimitDetail)
        XCTAssertEqual(
            snapshot.availability,
            .unavailable(
                reason: "missingEntitlement",
                detail: "PCC entitlement missing from signed app: com.apple.developer.private-cloud-compute"
            )
        )
    }
}
