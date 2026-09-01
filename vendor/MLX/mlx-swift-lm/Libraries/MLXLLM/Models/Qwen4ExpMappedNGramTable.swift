import Darwin
import Foundation
import MLX

enum Qwen4ExpMappedNGramTableError: Error, LocalizedError, Equatable {
    case cannotOpen(String)
    case cannotMap(String)
    case notRegularFile(String)
    case truncated
    case invalidHeader(String)
    case incompatible(expectedBits: Int, actualBits: Int, expectedGroupSize: Int, actualGroupSize: Int)
    case incompatibleGeometry(expectedRows: Int, actualRows: Int, expectedDimensions: Int, actualDimensions: Int)
    case rowOutOfRange(Int64)

    var errorDescription: String? {
        switch self {
        case .cannotOpen(let path):
            return "Cannot open Qwen n-gram table at \(path)"
        case .cannotMap(let path):
            return "Cannot memory-map Qwen n-gram table at \(path)"
        case .notRegularFile(let path):
            return "Qwen n-gram table is not a regular file: \(path)"
        case .truncated:
            return "Qwen n-gram table is truncated"
        case .invalidHeader(let detail):
            return "Invalid Qwen n-gram table header: \(detail)"
        case .incompatible(let expectedBits, let actualBits, let expectedGroupSize, let actualGroupSize):
            return "Qwen n-gram table quantization mismatch (expected \(expectedBits)-bit/group \(expectedGroupSize), got \(actualBits)-bit/group \(actualGroupSize))"
        case .incompatibleGeometry(let expectedRows, let actualRows, let expectedDimensions, let actualDimensions):
            return "Qwen n-gram table geometry mismatch (expected \(expectedRows)x\(expectedDimensions), got \(actualRows)x\(actualDimensions))"
        case .rowOutOfRange(let row):
            return "Qwen n-gram table row \(row) is out of range"
        }
    }
}

/// A read-only affine-quantized n-gram table that keeps the large sidecar on
/// disk. Decode gathers use parallel positional reads to avoid serial VM page
/// faults; larger prefill gathers use the mapped data directly.
final class Qwen4ExpMappedNGramTable: @unchecked Sendable {
    private final class MappedBuffer: @unchecked Sendable {
        let pointer: UnsafeRawPointer
        let count: Int

        init(descriptor: Int32, path: String) throws {
            var info = stat()
            guard Darwin.fstat(descriptor, &info) == 0 else {
                throw Qwen4ExpMappedNGramTableError.cannotOpen(path)
            }
            guard info.st_mode & S_IFMT == S_IFREG else {
                throw Qwen4ExpMappedNGramTableError.notRegularFile(path)
            }
            guard info.st_size > 0, info.st_size <= off_t(Int.max) else {
                throw Qwen4ExpMappedNGramTableError.truncated
            }
            count = Int(info.st_size)
            guard let mapping = Darwin.mmap(
                nil, count, PROT_READ, MAP_PRIVATE, descriptor, 0),
                mapping != MAP_FAILED
            else {
                throw Qwen4ExpMappedNGramTableError.cannotMap(path)
            }
            pointer = UnsafeRawPointer(mapping)
        }

        deinit {
            Darwin.munmap(UnsafeMutableRawPointer(mutating: pointer), count)
        }

        func withUnsafeBytes<R>(
            _ body: (UnsafeRawBufferPointer) throws -> R
        ) rethrows -> R {
            try body(UnsafeRawBufferPointer(start: pointer, count: count))
        }
    }

    private final class OutputBuffer: @unchecked Sendable {
        let pointer: UnsafeMutablePointer<UInt16>

        init(count: Int) {
            pointer = .allocate(capacity: count)
        }

        deinit {
            pointer.deallocate()
        }
    }

    private final class ByteBuffer: @unchecked Sendable {
        let pointer: UnsafeMutablePointer<UInt8>

        init(count: Int) {
            pointer = .allocate(capacity: count)
        }

        deinit {
            pointer.deallocate()
        }
    }

    /// Persistent positional-read workers for the decode-width gather. A
    /// global dispatch queue limits useful I/O parallelism and pays scheduling
    /// overhead on every token; this pool parks one worker per independent
    /// `(row, tensor-region)` read and exists only for an opened mapped table.
    private final class PositionalReadPool: @unchecked Sendable {
        private let workerCount = 48
        private let submissionLock = NSLock()
        private let condition = NSCondition()
        private let descriptor: Int32
        private let weightOffset: Int
        private let scaleOffset: Int
        private let biasOffset: Int
        private let weightBytesPerRow: Int
        private let scaleBytesPerRow: Int

        private var threads = [Thread]()
        private var generation = 0
        private var stopping = false
        private var stoppedWorkers = 0
        private var pendingWorkers = 0
        private var failed = false
        private var jobIDs = [Int64]()
        private var jobBuffer: ByteBuffer?

        init(
            descriptor: Int32,
            weightOffset: Int,
            scaleOffset: Int,
            biasOffset: Int,
            weightBytesPerRow: Int,
            scaleBytesPerRow: Int
        ) {
            self.descriptor = descriptor
            self.weightOffset = weightOffset
            self.scaleOffset = scaleOffset
            self.biasOffset = biasOffset
            self.weightBytesPerRow = weightBytesPerRow
            self.scaleBytesPerRow = scaleBytesPerRow

            threads.reserveCapacity(workerCount)
            for index in 0 ..< workerCount {
                let thread = Thread { [unowned self] in
                    self.worker(index: index)
                }
                thread.name = "afm-qwen-ngram-\(index)"
                thread.qualityOfService = .userInitiated
                threads.append(thread)
                thread.start()
            }
        }

        deinit {
            shutdown()
        }

        func run(_ ids: [Int64]) -> (buffer: ByteBuffer, succeeded: Bool) {
            submissionLock.lock()
            defer { submissionLock.unlock() }
            let rowBytes = weightBytesPerRow + 2 * scaleBytesPerRow
            let buffer = ByteBuffer(count: ids.count * rowBytes)

            condition.lock()
            precondition(pendingWorkers == 0 && !stopping)
            jobIDs = ids
            jobBuffer = buffer
            failed = false
            pendingWorkers = threads.count
            generation &+= 1
            condition.broadcast()
            while pendingWorkers > 0 {
                condition.wait()
            }
            let succeeded = !failed
            jobIDs.removeAll(keepingCapacity: true)
            jobBuffer = nil
            condition.unlock()
            return (buffer, succeeded)
        }

        func shutdown() {
            condition.lock()
            if !stopping {
                stopping = true
                condition.broadcast()
            }
            while stoppedWorkers < threads.count {
                condition.wait()
            }
            condition.unlock()
        }

        private func worker(index: Int) {
            var seenGeneration = 0
            while true {
                condition.lock()
                while generation == seenGeneration && !stopping {
                    condition.wait()
                }
                if stopping {
                    stoppedWorkers += 1
                    condition.broadcast()
                    condition.unlock()
                    return
                }
                seenGeneration = generation
                let ids = jobIDs
                let buffer = jobBuffer!
                condition.unlock()

                let rowBytes = weightBytesPerRow + 2 * scaleBytesPerRow
                var localFailure = false
                var site = index
                while site < ids.count * 3 {
                    let rowIndex = site / 3
                    let region = site % 3
                    let row = Int(ids[rowIndex])
                    let rowStart = buffer.pointer + rowIndex * rowBytes
                    let success: Bool
                    switch region {
                    case 0:
                        success = Qwen4ExpMappedNGramTable.readExactly(
                            descriptor,
                            into: rowStart,
                            count: weightBytesPerRow,
                            offset: weightOffset + row * weightBytesPerRow)
                    case 1:
                        success = Qwen4ExpMappedNGramTable.readExactly(
                            descriptor,
                            into: rowStart + weightBytesPerRow,
                            count: scaleBytesPerRow,
                            offset: scaleOffset + row * scaleBytesPerRow)
                    default:
                        success = Qwen4ExpMappedNGramTable.readExactly(
                            descriptor,
                            into: rowStart + weightBytesPerRow + scaleBytesPerRow,
                            count: scaleBytesPerRow,
                            offset: biasOffset + row * scaleBytesPerRow)
                    }
                    localFailure = localFailure || !success
                    site += workerCount
                }

                condition.lock()
                failed = failed || localFailure
                pendingWorkers -= 1
                if pendingWorkers == 0 {
                    condition.broadcast()
                }
                condition.unlock()
            }
        }
    }

    private struct TensorDescriptor: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]

        enum CodingKeys: String, CodingKey {
            case dtype
            case shape
            case dataOffsets = "data_offsets"
        }
    }

    private struct Header: Decodable {
        let metadata: [String: String]
        let weight: TensorDescriptor
        let scales: TensorDescriptor
        let biases: TensorDescriptor

        enum CodingKeys: String, CodingKey {
            case metadata = "__metadata__"
            case weight
            case scales
            case biases
        }
    }

    let rows: Int
    let dimensions: Int
    let bits: Int
    let groupSize: Int

    private let mappedData: MappedBuffer
    private let fileDescriptor: Int32
    private let weightOffset: Int
    private let scaleOffset: Int
    private let biasOffset: Int
    private let weightBytesPerRow: Int
    private let scaleBytesPerRow: Int
    private let parallelReadLimit = 64
    private var positionalReadPool: PositionalReadPool?

    init(
        url: URL,
        expectedRows: Int,
        expectedDimensions: Int,
        expectedBits: Int,
        expectedGroupSize: Int
    ) throws {
        let descriptor = url.withUnsafeFileSystemRepresentation { path in
            path.map { Darwin.open($0, O_RDONLY) } ?? -1
        }
        guard descriptor >= 0 else {
            throw Qwen4ExpMappedNGramTableError.cannotOpen(url.path)
        }

        do {
            // Map the same opened regular file used by positional reads. This
            // prevents a path replacement from mixing metadata and payloads,
            // and cannot silently fall back to a multi-gigabyte heap copy.
            let data = try MappedBuffer(descriptor: descriptor, path: url.path)
            guard data.count >= MemoryLayout<UInt64>.size else {
                throw Qwen4ExpMappedNGramTableError.truncated
            }
            let headerLength = data.withUnsafeBytes { raw -> UInt64 in
                raw.loadUnaligned(as: UInt64.self).littleEndian
            }
            let maximumHeaderBytes = 16 * 1_024 * 1_024
            guard headerLength <= UInt64(maximumHeaderBytes) else {
                throw Qwen4ExpMappedNGramTableError.truncated
            }
            let dataOffset = try Self.checkedAdd(8, Int(headerLength))
            guard dataOffset <= data.count else {
                throw Qwen4ExpMappedNGramTableError.truncated
            }
            let header: Header
            do {
                let headerData = data.withUnsafeBytes { raw in
                    Data(raw[8 ..< dataOffset])
                }
                header = try JSONDecoder().decode(Header.self, from: headerData)
            } catch {
                throw Qwen4ExpMappedNGramTableError.invalidHeader(String(describing: error))
            }

            guard header.metadata["format"] == "mlx-serve-ngram" else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader(
                    "unsupported or missing n-gram sidecar format")
            }
            guard let actualBits = header.metadata["bits"].flatMap(Int.init),
                  let actualGroupSize = header.metadata["group_size"].flatMap(Int.init),
                  actualBits == 4,
                  actualGroupSize > 0
            else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader(
                    "expected 4-bit affine quantization metadata")
            }
            guard actualBits == expectedBits, actualGroupSize == expectedGroupSize else {
                throw Qwen4ExpMappedNGramTableError.incompatible(
                    expectedBits: expectedBits,
                    actualBits: actualBits,
                    expectedGroupSize: expectedGroupSize,
                    actualGroupSize: actualGroupSize)
            }
            guard header.weight.dtype == "U32",
                  header.scales.dtype == "BF16",
                  header.biases.dtype == "BF16"
            else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader("expected U32 weight and BF16 scales/biases")
            }
            guard header.weight.shape.count == 2,
                  header.scales.shape.count == 2,
                  header.biases.shape.count == 2,
                  header.weight.dataOffsets.count == 2,
                  header.scales.dataOffsets.count == 2,
                  header.biases.dataOffsets.count == 2
            else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader("expected rank-2 tensors and two data offsets")
            }

            let actualRows = header.weight.shape[0]
            let packedColumns = header.weight.shape[1]
            let scaleColumns = header.scales.shape[1]
            let actualDimensions = try Self.checkedMultiply(scaleColumns, actualGroupSize)
            let logicalWeightBits = try Self.checkedMultiply(actualDimensions, actualBits)
            let packedWeightBits = try Self.checkedMultiply(packedColumns, 32)
            guard header.scales.shape[0] == actualRows,
                  header.biases.shape == header.scales.shape,
                  logicalWeightBits == packedWeightBits
            else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader("inconsistent tensor geometry")
            }
            guard actualRows == expectedRows, actualDimensions == expectedDimensions else {
                throw Qwen4ExpMappedNGramTableError.incompatibleGeometry(
                    expectedRows: expectedRows,
                    actualRows: actualRows,
                    expectedDimensions: expectedDimensions,
                    actualDimensions: actualDimensions)
            }

            let weightBytes = try Self.checkedMultiply(try Self.checkedMultiply(actualRows, packedColumns), 4)
            let scaleBytes = try Self.checkedMultiply(try Self.checkedMultiply(actualRows, scaleColumns), 2)
            try Self.validate(
                descriptor: header.weight,
                expectedBytes: weightBytes,
                dataOffset: dataOffset,
                fileSize: data.count,
                name: "weight")
            try Self.validate(
                descriptor: header.scales,
                expectedBytes: scaleBytes,
                dataOffset: dataOffset,
                fileSize: data.count,
                name: "scales")
            try Self.validate(
                descriptor: header.biases,
                expectedBytes: scaleBytes,
                dataOffset: dataOffset,
                fileSize: data.count,
                name: "biases")
            guard header.weight.dataOffsets[1] <= header.scales.dataOffsets[0],
                  header.scales.dataOffsets[1] <= header.biases.dataOffsets[0]
            else {
                throw Qwen4ExpMappedNGramTableError.invalidHeader(
                    "overlapping tensor byte ranges")
            }

            self.rows = actualRows
            self.dimensions = actualDimensions
            self.bits = actualBits
            self.groupSize = actualGroupSize
            self.mappedData = data
            self.fileDescriptor = descriptor
            self.weightOffset = try Self.checkedAdd(dataOffset, header.weight.dataOffsets[0])
            self.scaleOffset = try Self.checkedAdd(dataOffset, header.scales.dataOffsets[0])
            self.biasOffset = try Self.checkedAdd(dataOffset, header.biases.dataOffsets[0])
            self.weightBytesPerRow = try Self.checkedMultiply(packedColumns, 4)
            self.scaleBytesPerRow = try Self.checkedMultiply(scaleColumns, 2)
            self.positionalReadPool = PositionalReadPool(
                descriptor: descriptor,
                weightOffset: self.weightOffset,
                scaleOffset: self.scaleOffset,
                biasOffset: self.biasOffset,
                weightBytesPerRow: self.weightBytesPerRow,
                scaleBytesPerRow: self.scaleBytesPerRow)
        } catch {
            Darwin.close(descriptor)
            throw error
        }
    }

    deinit {
        positionalReadPool?.shutdown()
        positionalReadPool = nil
        Darwin.close(fileDescriptor)
    }

    func gather(_ rowIDs: MLXArray) throws -> MLXArray {
        let ids = rowIDs.asType(.int64).reshaped(-1).asArray(Int64.self)
        return try gather(ids, shape: rowIDs.shape)
    }

    /// Gather row IDs that were already computed on the host. The mapped PLE
    /// path computes its n-gram hashes from the generation token IDs on the
    /// CPU, so wrapping those IDs in an MLX array and synchronizing them back
    /// here only adds a needless host/device boundary to every decode token.
    func gather(_ ids: [Int64], shape: [Int]) throws -> MLXArray {
        guard let invalid = ids.first(where: { $0 < 0 || $0 >= Int64(rows) }) else {
            let count = try Self.checkedMultiply(ids.count, dimensions)
            let output = OutputBuffer(count: count)
            if ids.count <= parallelReadLimit {
                if !gatherWithPositionalReads(ids, output: output) {
                    // A regular-file pread can be interrupted or fail after the
                    // sidecar was opened. The already validated read-only mapping
                    // is the safe in-process fallback for this generation step.
                    gatherFromMapping(ids, output: output)
                }
            } else {
                gatherFromMapping(ids, output: output)
            }
            let data = Data(
                bytes: output.pointer,
                count: try Self.checkedMultiply(count, 2))
            return MLXArray(data, shape + [dimensions], dtype: .bfloat16)
        }
        throw Qwen4ExpMappedNGramTableError.rowOutOfRange(invalid)
    }

    private func gatherWithPositionalReads(
        _ ids: [Int64], output: OutputBuffer
    ) -> Bool {
        guard let positionalReadPool else { return false }
        let rowBytes = weightBytesPerRow + 2 * scaleBytesPerRow
        let result = positionalReadPool.run(ids)
        guard result.succeeded else { return false }
        let raw = UnsafeRawBufferPointer(
            start: result.buffer.pointer,
            count: ids.count * rowBytes)
        for index in ids.indices {
            let rowStart = index * rowBytes
            dequantize(
                weight: raw,
                weightStart: rowStart,
                scaleStart: rowStart + weightBytesPerRow,
                biasStart: rowStart + weightBytesPerRow + scaleBytesPerRow,
                output: output.pointer + index * dimensions)
        }
        return true
    }

    private func gatherFromMapping(
        _ ids: [Int64], output: OutputBuffer
    ) {
        mappedData.withUnsafeBytes { raw in
            for (index, rawRow) in ids.enumerated() {
                let row = Int(rawRow)
                dequantize(
                    weight: raw,
                    weightStart: weightOffset + row * weightBytesPerRow,
                    scaleStart: scaleOffset + row * scaleBytesPerRow,
                    biasStart: biasOffset + row * scaleBytesPerRow,
                    output: output.pointer + index * dimensions)
            }
        }
    }

    private func dequantize(
        weight: UnsafeRawBufferPointer,
        weightStart: Int,
        scaleStart: Int,
        biasStart: Int,
        output: UnsafeMutablePointer<UInt16>
    ) {
        let mask = UInt32((1 << bits) - 1)
        for column in 0 ..< dimensions {
            let bitOffset = column * bits
            let wordIndex = bitOffset / 32
            let shift = bitOffset % 32
            var packed = UInt64(Self.uint32(weight, at: weightStart + wordIndex * 4))
            if shift + bits > 32 {
                packed |= UInt64(Self.uint32(weight, at: weightStart + (wordIndex + 1) * 4)) << 32
            }
            let quantized = UInt32((packed >> UInt64(shift)) & UInt64(mask))
            let group = column / groupSize
            let scale = Self.bfloat16(weight, at: scaleStart + group * 2)
            let bias = Self.bfloat16(weight, at: biasStart + group * 2)
            output[column] = Self.toBFloat16(Float(quantized) * scale + bias)
        }
    }

    private static func validate(
        descriptor: TensorDescriptor,
        expectedBytes: Int,
        dataOffset: Int,
        fileSize: Int,
        name: String
    ) throws {
        let start = descriptor.dataOffsets[0]
        let end = descriptor.dataOffsets[1]
        guard start >= 0, end >= start, end - start == expectedBytes else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader("invalid \(name) byte range")
        }
        let absoluteEnd = try checkedAdd(dataOffset, end)
        guard absoluteEnd <= fileSize else {
            throw Qwen4ExpMappedNGramTableError.truncated
        }
    }

    private static func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader("integer overflow")
        }
        return value
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader("integer overflow")
        }
        return value
    }

    private static func readExactly(
        _ descriptor: Int32,
        into pointer: UnsafeMutableRawPointer,
        count: Int,
        offset: Int
    ) -> Bool {
        var completed = 0
        while completed < count {
            let result = Darwin.pread(
                descriptor,
                pointer + completed,
                count - completed,
                off_t(offset + completed))
            if result > 0 {
                completed += result
                continue
            }
            if result < 0, errno == EINTR {
                continue
            }
            return false
        }
        return true
    }

    private static func uint32(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
    }

    private static func bfloat16(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> Float {
        let value = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        return Float(bitPattern: UInt32(value) << 16)
    }

    private static func toBFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16).littleEndian
    }
}
