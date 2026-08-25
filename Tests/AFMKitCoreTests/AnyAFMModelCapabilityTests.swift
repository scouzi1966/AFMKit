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

        let admission = try await model.admissionSnapshot()
        XCTAssertEqual(admission.executionMode, .concurrent)
        XCTAssertEqual(admission.maximumConcurrentOperations, 4)

        let telemetry = try await model.telemetrySnapshot()
        XCTAssertEqual(telemetry.activeOperations, 1)
        XCTAssertEqual(telemetry.metadata["runtime"], .string("test"))
    }

    func testUnsupportedCapabilitiesFailExplicitly() async {
        let model = AnyAFMModel(BasicModel())
        let erased: any AFMModel = model

        XCTAssertFalse(model.supportsTokenization)
        XCTAssertFalse(model.supportsPrewarming)
        XCTAssertFalse(model.supportsAdmissionReporting)
        XCTAssertFalse(model.supportsTelemetryReporting)
        XCTAssertNil(erased as? any AFMTextTokenizing)
        XCTAssertNil(erased as? any AFMPrewarmableModel)
        XCTAssertNil(erased as? any AFMAdmissionReportingModel)
        XCTAssertNil(erased as? any AFMTelemetryReportingModel)

        await XCTAssertThrowsErrorAsync(try await model.tokenize(text: "hello")) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("tokenization"))
        }
        await XCTAssertThrowsErrorAsync(try await model.prewarm()) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("prewarming"))
        }
        await XCTAssertThrowsErrorAsync(try await model.admissionSnapshot()) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("admission reporting"))
        }
        await XCTAssertThrowsErrorAsync(try await model.telemetrySnapshot()) { error in
            XCTAssertEqual(error as? AFMError, .unsupportedCapability("telemetry reporting"))
        }
    }

    func testRepeatedTypeErasurePreservesActualCapabilities() async throws {
        let state = CapabilityState()
        let capable = AnyAFMModel(AnyAFMModel(CapableModel(state: state)))
        let basic = AnyAFMModel(AnyAFMModel(BasicModel()))

        XCTAssertTrue(capable.supportsTokenization)
        XCTAssertTrue(capable.supportsPrewarming)
        let tokens = try await capable.tokenize(text: "hello")
        XCTAssertEqual(tokens, [5])
        XCTAssertFalse(basic.supportsTokenization)
        XCTAssertFalse(basic.supportsPrewarming)
        XCTAssertTrue(capable.supportsAdmissionReporting)
        XCTAssertTrue(capable.supportsTelemetryReporting)
        XCTAssertFalse(basic.supportsAdmissionReporting)
        XCTAssertFalse(basic.supportsTelemetryReporting)
        XCTAssertNil((capable as any AFMModel) as? any AFMAdmissionReportingModel)
        XCTAssertNil((basic as any AFMModel) as? any AFMTelemetryReportingModel)
    }

    func testTypeErasureReadsDescriptorDynamicallyAfterLoad() async throws {
        let state = DescriptorState(testDescriptor(modelID: "declared"))
        let erased = AnyAFMModel(DynamicDescriptorModel(state: state))

        XCTAssertEqual(erased.descriptor.modelID, "declared")
        let loaded = try await erased.load()

        XCTAssertEqual(loaded.modelID, "qualified")
        XCTAssertEqual(erased.descriptor.modelID, "qualified")
        XCTAssertTrue(erased.descriptor.capabilities.contains(.vision))
    }
}

private final class DescriptorState: @unchecked Sendable {
    private let lock = NSLock()
    private var value: AFMModelDescriptor

    init(_ value: AFMModelDescriptor) { self.value = value }

    func read() -> AFMModelDescriptor { lock.withLock { value } }
    func write(_ descriptor: AFMModelDescriptor) { lock.withLock { value = descriptor } }
}

private struct DynamicDescriptorModel: AFMModel {
    let state: DescriptorState
    var descriptor: AFMModelDescriptor { state.read() }

    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        let qualified = AFMModelDescriptor(
            providerID: "test",
            modelID: "qualified",
            displayName: "qualified",
            capabilities: [.text, .vision],
            privacyBoundary: .device
        )
        state.write(qualified)
        return qualified
    }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        AFMModelResponse(text: "")
    }
    func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { $0.finish() }
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

private struct CapableModel: AFMModel, AFMTextTokenizing, AFMPrewarmableModel,
    AFMAdmissionReportingModel, AFMTelemetryReportingModel
{
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
    func admissionSnapshot() async -> AFMAdmissionSnapshot {
        AFMAdmissionSnapshot(
            executionMode: .concurrent,
            maximumConcurrentOperations: 4,
            activeOperations: 1,
            queuedOperations: 0,
            availableOperationSlots: 3
        )
    }
    func telemetrySnapshot() async -> AFMTelemetrySnapshot {
        AFMTelemetrySnapshot(
            activeOperations: 1,
            peakMemoryGib: 2.5,
            metadata: ["runtime": .string("test")]
        )
    }
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
