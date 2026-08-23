import XCTest
import AFMKitServices

final class AFMKitServicesTests: XCTestCase {
    func testUmbrellaReexportsEveryOptionalService() {
        _ = VisionService()
        _ = SpeechService()
        _ = SpeechSynthesisService()
        _ = EmbeddingModelRegistry()
    }
}
