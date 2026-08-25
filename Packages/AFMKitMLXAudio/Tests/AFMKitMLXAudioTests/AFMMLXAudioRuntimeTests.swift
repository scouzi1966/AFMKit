import XCTest
import AFMKitMLX
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon
@testable import AFMKitMLXAudio

final class AFMMLXAudioRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

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

    func testSuccessfulLoadAndSynthesisReturnRuntimeToReady() async throws {
        let fixture = try makeFixture(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let fake = FakeSpeechModel(samples: [0.25, -0.5], sampleRate: 24_000)
        let runtime = AFMMLXAudioRuntime(modelStore: fixture.store) { _, _, downloadIfNeeded in
            XCTAssertFalse(downloadIfNeeded)
            return fake
        }

        let descriptor = try await runtime.load(modelID: fixture.model.path, family: .qwen3TTS)
        let result = try await runtime.synthesize(.init(text: "Hello"))

        XCTAssertEqual(descriptor.sampleRate, 24_000)
        XCTAssertEqual(result.samples, [0.25, -0.5])
        XCTAssertEqual(result.sampleRate, 24_000)
        let finalState = await runtime.state
        XCTAssertEqual(finalState, .ready(descriptor))
    }

    func testStreamingMapsEventsAndReturnsRuntimeToReady() async throws {
        let fixture = try makeFixture(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let info = AudioGenerationInfo(
            promptTokenCount: 2,
            generationTokenCount: 3,
            prefillTime: 0.1,
            generateTime: 0.2,
            tokensPerSecond: 15,
            peakMemoryUsage: 0.5
        )
        let fake = FakeSpeechModel(
            streamEvents: [.token(7), .audio(MLXArray([Float(0.5)])), .info(info)]
        )
        let runtime = AFMMLXAudioRuntime(modelStore: fixture.store) { _, _, _ in fake }
        let descriptor = try await runtime.load(modelID: fixture.model.path, family: .qwen3TTS)

        let stream = try await runtime.stream(.init(text: "Stream"))
        var events: [AFMMLXAudioStreamEvent] = []
        for try await event in stream { events.append(event) }

        XCTAssertEqual(events.count, 4)
        XCTAssertEqual(events[0], .token(7))
        XCTAssertEqual(events[1], .audio(samples: [0.5], sampleRate: 24_000))
        if case .metrics(let metrics) = events[2] {
            XCTAssertEqual(metrics.generationTokenCount, 3)
        } else {
            XCTFail("Expected metrics event")
        }
        XCTAssertEqual(events[3], .completed)
        let finalState = await runtime.state
        XCTAssertEqual(finalState, .ready(descriptor))
    }

    func testSynthesisCancellationReachesProducerAndReturnsReady() async throws {
        let fixture = try makeFixture(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let probe = OperationProbe()
        let fake = FakeSpeechModel(generationDelay: .seconds(30), generationProbe: probe)
        let runtime = AFMMLXAudioRuntime(modelStore: fixture.store) { _, _, _ in fake }
        let descriptor = try await runtime.load(modelID: fixture.model.path, family: .qwen3TTS)
        let generation = Task { try await runtime.synthesize(.init(text: "Cancel")) }
        await probe.waitUntilEntered()

        await runtime.cancelActiveGeneration()
        do {
            _ = try await generation.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {}

        let wasCancelled = await probe.wasCancelled
        let finalState = await runtime.state
        XCTAssertTrue(wasCancelled)
        XCTAssertEqual(finalState, .ready(descriptor))
    }

    func testStreamingCancellationReachesProducerAndReturnsReady() async throws {
        let fixture = try makeFixture(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let cancellation = expectation(description: "upstream stream producer cancelled")
        let probe = OperationProbe(cancellationExpectation: cancellation)
        let fake = FakeSpeechModel(streamDelay: .seconds(30), streamProbe: probe)
        let runtime = AFMMLXAudioRuntime(modelStore: fixture.store) { _, _, _ in fake }
        let descriptor = try await runtime.load(modelID: fixture.model.path, family: .qwen3TTS)
        let stream = try await runtime.stream(.init(text: "Cancel stream"))
        let consumer = Task {
            for try await _ in stream {}
        }
        await probe.waitUntilEntered()

        await runtime.cancelActiveGeneration()
        _ = try? await consumer.value

        await fulfillment(of: [cancellation], timeout: 2)
        let wasCancelled = await probe.wasCancelled
        let finalState = await runtime.state
        XCTAssertTrue(wasCancelled)
        XCTAssertEqual(finalState, .ready(descriptor))
    }

    func testSupersededLoadCannotReplaceNewerModel() async throws {
        let first = try makeFixture(modelType: "qwen3_tts", name: "first")
        defer { try? FileManager.default.removeItem(at: first.root) }
        let secondModel = try makeModelDirectory(root: first.root, modelType: "qwen3_tts", name: "second")
        let firstFake = FakeSpeechModel(sampleRate: 16_000)
        let secondFake = FakeSpeechModel(sampleRate: 24_000)
        let firstLoad = SuspensionGate()
        let runtime = AFMMLXAudioRuntime(modelStore: first.store) { location, _, _ in
            if location.modelDirectory.lastPathComponent == "first" {
                await firstLoad.pause()
                return firstFake
            }
            return secondFake
        }
        let stale = Task {
            try await runtime.load(modelID: first.model.path, family: .qwen3TTS)
        }
        await firstLoad.waitUntilEntered()

        let current = try await runtime.load(modelID: secondModel.path, family: .qwen3TTS)
        await firstLoad.release()
        do {
            _ = try await stale.value
            XCTFail("Expected stale load rejection")
        } catch {
            XCTAssertEqual(error as? AFMMLXAudioError, .loadSuperseded)
        }

        XCTAssertEqual(current.modelID, secondModel.path)
        XCTAssertEqual(current.sampleRate, 24_000)
        let loadedModel = await runtime.loadedModel
        XCTAssertEqual(loadedModel, current)
    }

    func testUnloadDuringPendingLoadStaysUnloaded() async throws {
        let fixture = try makeFixture(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let fake = FakeSpeechModel()
        let pendingLoad = SuspensionGate()
        let runtime = AFMMLXAudioRuntime(modelStore: fixture.store) { _, _, _ in
            await pendingLoad.pause()
            return fake
        }
        let pending = Task {
            try await runtime.load(modelID: fixture.model.path, family: .qwen3TTS)
        }
        await pendingLoad.waitUntilEntered()

        await runtime.unload()
        await pendingLoad.release()
        do {
            _ = try await pending.value
            XCTFail("Expected pending load rejection")
        } catch {
            XCTAssertEqual(error as? AFMMLXAudioError, .loadSuperseded)
        }
        let finalState = await runtime.state
        let loadedModel = await runtime.loadedModel
        XCTAssertEqual(finalState, .unloaded)
        XCTAssertNil(loadedModel)
    }

    func testReplacementFailureClearsPreviousModel() async throws {
        enum Expected: Error { case load }
        let first = try makeFixture(modelType: "qwen3_tts", name: "first")
        defer { try? FileManager.default.removeItem(at: first.root) }
        let failingModel = try makeModelDirectory(root: first.root, modelType: "qwen3_tts", name: "failing")
        let fake = FakeSpeechModel()
        let runtime = AFMMLXAudioRuntime(modelStore: first.store) { location, _, _ in
            if location.modelDirectory.lastPathComponent == "failing" { throw Expected.load }
            return fake
        }
        _ = try await runtime.load(modelID: first.model.path, family: .qwen3TTS)

        do {
            _ = try await runtime.load(modelID: failingModel.path, family: .qwen3TTS)
            XCTFail("Expected replacement failure")
        } catch {}
        let loadedModel = await runtime.loadedModel
        XCTAssertNil(loadedModel)
        do {
            _ = try await runtime.synthesize(.init(text: "No stale model"))
            XCTFail("Expected model-not-loaded failure")
        } catch {
            XCTAssertEqual(error as? AFMMLXAudioError, .modelNotLoaded)
        }
    }

    private func makeFixture(
        modelType: String,
        name: String = "model"
    ) throws -> (root: URL, model: URL, store: AFMMLXAudioModelStore) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMMLXAudioRuntimeTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let model = try makeModelDirectory(root: root, modelType: modelType, name: name)
        let store = AFMMLXAudioModelStore(
            modelStore: .init(resolver: .init(cacheRoot: root)),
            importCacheDirectory: root.appendingPathComponent("overlay", isDirectory: true)
        )
        return (root, model, store)
    }

    private func makeModelDirectory(root: URL, modelType: String, name: String) throws -> URL {
        let model = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["model_type": modelType])
            .write(to: model.appendingPathComponent("config.json"))
        try Data([1]).write(to: model.appendingPathComponent("model.safetensors"))
        return model
    }
}

private actor OperationProbe {
    private var entered = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private(set) var wasCancelled = false
    private let cancellationExpectation: XCTestExpectation?

    init(cancellationExpectation: XCTestExpectation? = nil) {
        self.cancellationExpectation = cancellationExpectation
    }

    func markEntered() {
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func recordCancellation() {
        wasCancelled = true
        cancellationExpectation?.fulfill()
    }
}

private actor SuspensionGate {
    private var entered = false
    private var released = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func pause() async {
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        waiters.forEach { $0.resume() }
        if released { return }
        await withCheckedContinuation { continuation in
            releaseWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }
}

private final class FakeSpeechModel: @unchecked Sendable, SpeechGenerationModel {
    let sampleRate: Int
    let defaultGenerationParameters = GenerateParameters()
    private let samples: [Float]
    private let streamEvents: [AudioGeneration]
    private let generationDelay: Duration?
    private let streamDelay: Duration?
    private let generationProbe: OperationProbe?
    private let streamProbe: OperationProbe?

    init(
        samples: [Float] = [0],
        sampleRate: Int = 24_000,
        streamEvents: [AudioGeneration] = [],
        generationDelay: Duration? = nil,
        streamDelay: Duration? = nil,
        generationProbe: OperationProbe? = nil,
        streamProbe: OperationProbe? = nil
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.streamEvents = streamEvents
        self.generationDelay = generationDelay
        self.streamDelay = streamDelay
        self.generationProbe = generationProbe
        self.streamProbe = streamProbe
    }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        do {
            await generationProbe?.markEntered()
            if let generationDelay { try await Task.sleep(for: generationDelay) }
            try Task.checkCancellation()
            return MLXArray(samples)
        } catch is CancellationError {
            await generationProbe?.recordCancellation()
            throw CancellationError()
        }
    }

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let events = streamEvents
        let delay = streamDelay
        let probe = streamProbe
        return AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    await probe?.markEntered()
                    if let delay { try await Task.sleep(for: delay) }
                    for event in events { continuation.yield(event) }
                    continuation.finish()
                } catch is CancellationError {
                    await probe?.recordCancellation()
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
        }
    }
}
