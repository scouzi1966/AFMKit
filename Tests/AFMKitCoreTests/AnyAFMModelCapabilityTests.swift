@testable import AFMKitCore
import XCTest

final class AnyAFMModelCapabilityTests: XCTestCase {
    func testTypeErasurePreservesTokenizationAndPrewarming() async throws {
        let state = CapabilityState()
        let model = AnyAFMModel(CapableModel(state: state))

        XCTAssertTrue(model.supportsTokenization)
        XCTAssertTrue(model.supportsPrewarming)
        let tokens = try await model.tokenize(text: "hello")
        XCTAssertEqual(tokens, [5])

        try await model.prewarm()
        let prewarmCount = await state.prewarmCount()
        XCTAssertEqual(prewarmCount, 1)
    }

    func testUnsupportedCapabilitiesFailExplicitly() async {
        let model = AnyAFMModel(BasicModel())

        XCTAssertFalse(model.supportsTokenization)
        XCTAssertFalse(model.supportsPrewarming)

        await XCTAssertThrowsErrorAsync(try await model.tokenize(text: "hello")) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("tokenization"))
        }
        await XCTAssertThrowsErrorAsync(try await model.prewarm()) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("prewarming"))
        }
    }
}

private actor CapabilityState {
    private var count = 0

    func markPrewarmed() {
        count += 1
    }

    func prewarmCount() -> Int {
        count
    }
}

private struct CapableModel: AFMModel, AFMTextTokenizing, AFMPrewarmableModel {
    let state: CapabilityState
    let descriptor = testDescriptor(modelID: "capable")

    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        progress?(1)
        return descriptor
    }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        AFMModelResponse(text: "")
    }
    func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
    func tokenize(text: String) async throws -> [Int] { [text.count] }
    func prewarm() async throws { await state.markPrewarmed() }
}

private struct BasicModel: AFMModel {
    let descriptor = testDescriptor(modelID: "basic")

    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        descriptor
    }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        AFMModelResponse(text: "")
    }
    func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

private func testDescriptor(modelID: AFMModelID) -> AFMModelDescriptor {
    AFMModelDescriptor(
        providerID: "test",
        modelID: modelID,
        displayName: modelID.rawValue,
        capabilities: [.text],
        privacyBoundary: .device
    )
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ errorHandler: (Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail("Expected expression to throw")
    } catch {
        errorHandler(error)
    }
}
