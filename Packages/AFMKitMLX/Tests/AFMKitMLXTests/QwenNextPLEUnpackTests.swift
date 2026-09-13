import Foundation
@testable import MLXLLM
import XCTest

final class QwenNextPLEUnpackTests: XCTestCase {
    private func scalar(
        _ raw: UnsafeRawBufferPointer, weight: Int, scale: Int, bias: Int,
        dimensions: Int, group: Int, output: UnsafeMutablePointer<UInt16>
    ) {
        func coefficient(_ offset: Int) -> Float {
            Float(bitPattern: (UInt32(raw[offset]) | UInt32(raw[offset + 1]) << 8) << 16)
        }
        for column in 0..<dimensions {
            let q = (raw[weight + column / 2] >> ((column % 2) * 4)) & 15
            let offset = column / group * 2
            let value = Float(q) * coefficient(scale + offset) + coefficient(bias + offset)
            let bits = value.bitPattern
            output[column] = UInt16(truncatingIfNeeded:
                (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
        }
    }

    func testSIMDUnpackIsByteIdenticalIncludingUnalignedRowsAndExtremeCoefficients() {
        let coefficients: [UInt16] = [0, 0x8000, 1, 0x8001, 0x3b80, 0x3f81,
                                       0xbf83, 0x7f7f, 0xff7f, 0x7f80, 0xff80]
        for (dimensions, group) in [(160, 32), (256, 64), (128, 128), (32, 8)] {
            let weightStart = 3
            let scaleStart = weightStart + dimensions / 2 + 2
            let biasStart = scaleStart + dimensions / group * 2 + 3
            for index in coefficients.indices {
                var bytes = [UInt8](repeating: 0, count: biasStart + dimensions / group * 2)
                for byte in 0..<(dimensions / 2) {
                    bytes[weightStart + byte] = UInt8(truncatingIfNeeded: byte * 37 + index)
                }
                for g in 0..<(dimensions / group) {
                    for (start, bits) in [(scaleStart, coefficients[index]),
                                          (biasStart, coefficients[(index + g + 1) % coefficients.count])] {
                        bytes[start + g * 2] = UInt8(truncatingIfNeeded: bits)
                        bytes[start + g * 2 + 1] = UInt8(truncatingIfNeeded: bits >> 8)
                    }
                }
                var expected = [UInt16](repeating: 0, count: dimensions)
                var actual = [UInt16](repeating: 0x1234, count: dimensions + 2)
                bytes.withUnsafeBytes { raw in
                    expected.withUnsafeMutableBufferPointer {
                        scalar(raw, weight: weightStart, scale: scaleStart, bias: biasStart,
                               dimensions: dimensions, group: group, output: $0.baseAddress!)
                    }
                    actual.withUnsafeMutableBufferPointer {
                        XCTAssertTrue(Qwen4ExpMappedNGramTable.dequantizeAffine4Row(
                            weight: raw, weightStart: weightStart, scaleStart: scaleStart,
                            biasStart: biasStart, dimensions: dimensions, groupSize: group,
                            output: $0.baseAddress! + 1))
                    }
                }
                XCTAssertEqual(Array(actual[1...dimensions]), expected,
                               "dimensions=\(dimensions), group=\(group), coefficient=\(index)")
                XCTAssertEqual(actual.first, 0x1234)
                XCTAssertEqual(actual.last, 0x1234)
            }
        }
    }

    func testUnsupportedVectorGeometryDoesNotWrite() {
        let bytes = [UInt8](repeating: 0, count: 512)
        var output = [UInt16](repeating: 0x1234, count: 256)
        bytes.withUnsafeBytes { raw in
            output.withUnsafeMutableBufferPointer { buffer in
                for (dimensions, group) in [(0, 32), (160, 0), (160, 3), (159, 32)] {
                    XCTAssertFalse(Qwen4ExpMappedNGramTable.dequantizeAffine4Row(
                        weight: raw, weightStart: 0, scaleStart: 256, biasStart: 384,
                        dimensions: dimensions, groupSize: group, output: buffer.baseAddress!))
                }
            }
        }
        XCTAssertTrue(output.allSatisfy { $0 == 0x1234 })
    }

    func testOptionalWarmRowMicrobenchmark() throws {
        guard ProcessInfo.processInfo.environment["AFM_PLE_UNPACK_MICROBENCH"] == "1"
        else { throw XCTSkip("Opt-in CPU row-unpack measurement") }
        let dimensions = 160, group = 32
        var bytes = (0..<100).map { UInt8(truncatingIfNeeded: $0 * 37) }
        for offset in stride(from: 80, to: 100, by: 2) {
            bytes[offset] = 0x80; bytes[offset + 1] = 0x3b
        }
        var output = [UInt16](repeating: 0, count: dimensions)
        var times = [[Double](), [Double]()]
        bytes.withUnsafeBytes { raw in
            output.withUnsafeMutableBufferPointer { buffer in
                for trial in 0..<12 {
                    for mode in (trial.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0..<10_000 {
                            if mode == 0 {
                                scalar(raw, weight: 0, scale: 80, bias: 90,
                                       dimensions: dimensions, group: group, output: buffer.baseAddress!)
                            } else {
                                _ = Qwen4ExpMappedNGramTable.dequantizeAffine4Row(
                                    weight: raw, weightStart: 0, scaleStart: 80, biasStart: 90,
                                    dimensions: dimensions, groupSize: group, output: buffer.baseAddress!)
                            }
                        }
                        if trial >= 2 {
                            times[mode].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 10_000)
                        }
                    }
                }
            }
        }
        let medians = times.map { values in
            let ordered = values.sorted()
            return (ordered[4] + ordered[5]) / 2
        }
        XCTAssertNotEqual(output, Array(repeating: 0, count: dimensions))
        print("[PLE-unpack] warm row scalar_ns=\(medians[0]) SIMD_ns=\(medians[1])")
    }
}
