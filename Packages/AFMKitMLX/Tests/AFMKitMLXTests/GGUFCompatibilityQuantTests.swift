import Darwin
import GGUFLib
import Testing

@Suite("GGUF experimental compatibility quants")
struct GGUFCompatibilityQuantTests {
    @Test("faithfully decodes legacy Q5 blocks")
    func decodesLegacyQ5() throws {
        var q50 = [UInt8](repeating: 0, count: 22)
        q50[0] = 0x00
        q50[1] = 0x3c // FP16 1.0
        q50[2 ..< 6] = [0xff, 0xff, 0xff, 0xff]
        for index in 0 ..< 16 {
            q50[6 + index] = UInt8(index | ((15 - index) << 4))
        }
        let decodedQ50 = try decode(type: UInt32(GGUF_TYPE_Q5_0.rawValue), bytes: q50, count: 32)
        let expectedQ50 = (0 ..< 16).map(Float.init) + (0 ..< 16).reversed().map(Float.init)
        #expect(decodedQ50 == expectedQ50)

        var q51 = [UInt8](repeating: 0, count: 24)
        q51[0] = 0x00
        q51[1] = 0x3c // FP16 scale 1.0
        q51[2] = 0x00
        q51[3] = 0x40 // FP16 minimum 2.0
        q51[4 ..< 8] = [0xff, 0xff, 0xff, 0xff]
        for index in 0 ..< 16 {
            q51[8 + index] = UInt8(index | ((15 - index) << 4))
        }
        let decodedQ51 = try decode(type: UInt32(GGUF_TYPE_Q5_1.rawValue), bytes: q51, count: 32)
        let expectedQ51 = (0 ..< 16).map { Float($0 + 18) } + (0 ..< 16).reversed().map { Float($0 + 18) }
        #expect(decodedQ51 == expectedQ51)
    }

    @Test("faithfully decodes Q3_K, Q5_K, and Q8_K blocks")
    func decodesKQuants() throws {
        var q3 = [UInt8](repeating: 0, count: 110)
        q3.replaceSubrange(0 ..< 32, with: repeatElement(UInt8(0xff), count: 32))
        q3.replaceSubrange(32 ..< 96, with: repeatElement(UInt8(0xe4), count: 64))
        q3[108] = 0x00
        q3[109] = 0x3c // FP16 super-scale 1.0
        let decodedQ3 = try decode(type: UInt32(GGUF_TYPE_Q3_K.rawValue), bytes: q3, count: 256)
        let expectedQ3Group: [Float] = [0, -32, -64, -96].flatMap { Array(repeating: $0, count: 32) }
        #expect(decodedQ3 == expectedQ3Group + expectedQ3Group)

        var q5 = [UInt8](repeating: 0, count: 176)
        q5[0] = 0x00
        q5[1] = 0x3c // FP16 scale-of-scales 1.0
        q5[2] = 0x00
        q5[3] = 0x3c // FP16 scale-of-minimums 1.0
        q5[4 ..< 8] = [1, 1, 1, 1]
        q5[12 ..< 16] = [1, 1, 1, 1]
        q5.replaceSubrange(16 ..< 48, with: repeatElement(UInt8(0xff), count: 32))
        q5.replaceSubrange(48 ..< 176, with: repeatElement(UInt8(0x10), count: 128))
        let decodedQ5 = try decode(type: UInt32(GGUF_TYPE_Q5_K.rawValue), bytes: q5, count: 256)
        let expectedQ5Pair = Array(repeating: Float(16), count: 32) + Array(repeating: Float(17), count: 32)
        #expect(decodedQ5 == Array(repeating: expectedQ5Pair, count: 4).flatMap { $0 })

        var q8 = [UInt8](repeating: 0, count: 292)
        var scale = Float(0.5)
        withUnsafeBytes(of: &scale) { q8.replaceSubrange(0 ..< 4, with: $0) }
        for index in 0 ..< 256 {
            q8[4 + index] = UInt8(bitPattern: Int8(truncatingIfNeeded: index - 128))
        }
        let decodedQ8 = try decode(type: UInt32(GGUF_TYPE_Q8_K.rawValue), bytes: q8, count: 256)
        #expect(decodedQ8 == (0 ..< 256).map { Float(Int8(truncatingIfNeeded: $0 - 128)) * 0.5 })
    }

    private func decode(type: UInt32, bytes: [UInt8], count: Int) throws -> [Float] {
        try bytes.withUnsafeBytes { rawBytes in
            var tensor = gguf_tensor()
            tensor.type = type
            tensor.num_weights = UInt64(count)
            tensor.weights_data = UnsafeMutablePointer(mutating: rawBytes.bindMemory(to: UInt8.self).baseAddress!)
            guard let decoded = gguf_tensor_to_float(&tensor) else {
                throw DecodeError.failed(type)
            }
            defer { free(decoded) }
            return Array(UnsafeBufferPointer(start: decoded, count: count))
        }
    }

    private enum DecodeError: Error {
        case failed(UInt32)
    }
}
