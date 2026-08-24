import XCTest
@testable import AFMKitMLXAudio

final class AFMMLXAudioRuntimeTests: XCTestCase {
    func testRuntimeStartsUnloaded() async {
        let runtime = AFMMLXAudioRuntime()
        let state = await runtime.state
        let model = await runtime.loadedModel

        XCTAssertEqual(state, .unloaded)
        XCTAssertNil(model)
    }

    func testRuntimeRejectsEmptyModelIDWithoutDownloading() async {
        let runtime = AFMMLXAudioRuntime()

        do {
            try await runtime.load(modelID: "   ")
            XCTFail("Expected empty model ID failure")
        } catch {
            XCTAssertEqual(error as? AFMMLXAudioError, .emptyModelID)
        }
    }

    func testRuntimeRejectsGenerationBeforeLoad() async {
        let runtime = AFMMLXAudioRuntime()

        do {
            _ = try await runtime.synthesize(.init(text: "Hello"))
            XCTFail("Expected model-not-loaded failure")
        } catch {
            XCTAssertEqual(error as? AFMMLXAudioError, .modelNotLoaded)
        }
    }

    func testUnloadIsIdempotent() async {
        let runtime = AFMMLXAudioRuntime()
        await runtime.unload()
        await runtime.unload()

        let state = await runtime.state
        let model = await runtime.loadedModel
        XCTAssertEqual(state, .unloaded)
        XCTAssertNil(model)
    }
}
