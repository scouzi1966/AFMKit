import MLX
import MLXNN
import MLXLMCommon
import XCTest

final class GenerationTokenCountTests: XCTestCase {
    private final class FixedSequenceModel: Module, LanguageModel {
        let sequence: [Int]
        var position = 0

        init(_ sequence: [Int]) { self.sequence = sequence }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let token = sequence[min(position, sequence.count - 1)]
            position += 1
            var logits = Array(repeating: Float(-100), count: 8)
            logits[token] = 100
            return MLXArray(logits).reshaped(1, 1, 8)
        }
    }

    private func assertCount(_ tokens: [Int], limit: Int, expected: Int, text: String) async throws {
        var tokenizer = TestTokenizer(eosTokenId: 0)
        tokenizer.byteVocabulary = [1: [65], 2: [66], 3: [0xE2], 4: [0x82], 5: [0xAC]]
        let model = FixedSequenceModel(tokens)
        let iterator = try TokenIterator(input: LMInput(text: .init(tokens: MLXArray([7]))),
            model: model, cache: [], parameters: GenerateParameters(maxTokens: limit, temperature: 0))
        let (stream, task) = generateTask(promptTokenCount: 1,
            modelConfiguration: ModelConfiguration(id: "fixture", toolCallFormat: ToolCallFormat.none),
            tokenizer: tokenizer, iterator: iterator)
        var output = ""
        var count: Int?
        for await event in stream {
            if case .chunk(let chunk) = event { output += chunk }
            if case .info(let info) = event { count = info.generationTokenCount }
        }
        await task.value
        XCTAssertEqual(count, expected)
        XCTAssertEqual(output, text)
    }

    func testASCIIExcludesEOS() async throws {
        try await assertCount([1, 2, 0], limit: 8, expected: 2, text: "AB")
    }

    func testBufferedUnicodeCountsEveryAcceptedID() async throws {
        try await assertCount([3, 4, 5, 0], limit: 8, expected: 3, text: "€")
    }

    func testTruncatedUnicodeCountsAcceptedUndeliveredIDs() async throws {
        try await assertCount([3, 4, 5, 0], limit: 2, expected: 2, text: "")
    }
}
