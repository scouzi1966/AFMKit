import Foundation
import Testing
@testable import AFMKitMLX

@Suite("GGUF loading")
struct AFMMLXGGUFLoaderTests {
    @Test("rejects non-GGUF paths before loading")
    func rejectsNonGGUFPath() {
        #expect(throws: Error.self) {
            _ = try AFMMLXGGUFLoader.load(url: URL(fileURLWithPath: "/tmp/model.bin"))
        }
    }

    @Test("loads an opt-in integration checkpoint")
    func loadsIntegrationCheckpoint() throws {
        guard let path = ProcessInfo.processInfo.environment["AFMKIT_GGUF_INTEGRATION_MODEL"] else {
            return
        }
        let checkpoint = try AFMMLXGGUFLoader.load(url: URL(fileURLWithPath: path))
        #expect(checkpoint.tensorCount > 0)
        #expect(checkpoint.totalElementCount > 0)
        #expect(checkpoint.architecture != nil)
        #expect(checkpoint.metadataKeys.contains("general.architecture"))
    }
}
