#if canImport(FoundationModels)
import AFMKitCore
@testable import AFMKitApple
import FoundationModels
import XCTest

@available(macOS 27.0, *)
final class FoundationModelsPortableAdapterTests: XCTestCase {
    func testPortableConfigurationContractLivesInAppleProduct() {
        struct Configuration: AFMFoundationModelsModelConfiguration {
            let defaultMaximumResponseTokens = 512
            let supportsReasoning = true
        }

        let configuration = Configuration()
        XCTAssertEqual(configuration.defaultMaximumResponseTokens, 512)
        XCTAssertTrue(configuration.supportsReasoning)
    }

    func testEventBridgeDoesNotPreloadModelBeforeStreaming() async throws {
        let state = InvocationState()
        let model = AnyAFMModel(ProbeModel(state: state))
        let request = AFMRequest(messages: [AFMMessage(role: .user, text: "Hello")])

        for try await _ in AFMFoundationModelsExecutorBridge.events(
            from: model,
            request: request
        ) {}

        let counts = await state.counts()
        XCTAssertEqual(counts.loads, 0)
        XCTAssertEqual(counts.streams, 1)
    }
}

private actor InvocationState {
    private var loadCount = 0
    private var streamCount = 0

    func recordLoad() { loadCount += 1 }
    func recordStream() { streamCount += 1 }
    func counts() -> (loads: Int, streams: Int) { (loadCount, streamCount) }
}

private struct ProbeModel: AFMModel {
    let state: InvocationState
    let descriptor = AFMModelDescriptor(
        providerID: "probe",
        modelID: "probe",
        displayName: "Probe",
        capabilities: [.text, .streaming],
        privacyBoundary: .device,
        requiresNetwork: false
    )

    func availability() async -> AFMModelAvailability { .available }

    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        await state.recordLoad()
        return descriptor
    }

    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        AFMModelResponse(text: "")
    }

    func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                await state.recordStream()
                continuation.yield(.completed(.stop))
                continuation.finish()
            }
        }
    }
}
#endif
