import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

final class DeepseekV4ArchitectureTests: XCTestCase {
    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    func testFusedHCCollapsePreservesBF16StorageWithFP32Parity() {
        let fixture = makeFixture()
        let residualBF16 = fixture.residual.asType(.bfloat16)

        let stored = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residualBF16,
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let fp32 = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residualBF16.asType(.float32),
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        MLX.eval(stored.collapsed, stored.post, stored.comb, fp32.collapsed)

        XCTAssertEqual(stored.collapsed.dtype, .bfloat16)
        XCTAssertEqual(fp32.collapsed.dtype, .float32)
        XCTAssertTrue(
            allClose(
                stored.collapsed,
                fp32.collapsed.asType(.bfloat16),
                rtol: 0,
                atol: 0).item(Bool.self))
        XCTAssertTrue(allClose(stored.post, fp32.post, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(allClose(stored.comb, fp32.comb, rtol: 0, atol: 0).item(Bool.self))
    }

    func testFusedHCExpandPreservesBF16StorageWithFP32Parity() {
        let fixture = makeFixture()
        let residualBF16 = fixture.residual.asType(.bfloat16)
        let collapsed = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residualBF16,
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let blockBF16 = MLXArray(
            (0..<fixture.hiddenSize).map { Float(($0 % 11) - 5) / 16 },
            [1, 1, fixture.hiddenSize]).asType(.bfloat16)

        let stored = DeepseekV4Math.hcExpand4(
            blockOut: blockBF16,
            residual: residualBF16,
            post: collapsed.post,
            comb: collapsed.comb,
            hiddenSize: fixture.hiddenSize)
        let fp32 = DeepseekV4Math.hcExpand4(
            blockOut: blockBF16.asType(.float32),
            residual: residualBF16.asType(.float32),
            post: collapsed.post,
            comb: collapsed.comb,
            hiddenSize: fixture.hiddenSize)
        MLX.eval(stored, fp32)

        XCTAssertEqual(stored.dtype, .bfloat16)
        XCTAssertEqual(fp32.dtype, .float32)
        XCTAssertTrue(
            allClose(stored, fp32.asType(.bfloat16), rtol: 0, atol: 0)
                .item(Bool.self))
    }

    func testFusedHCStoragePreservesFP16Parity() {
        assertFusedStorageParity(dtype: .float16, hiddenSize: 32)
    }

    func testFusedHCStoragePreservesBF16ParityAcrossMultipleThreadIterations() {
        assertFusedStorageParity(dtype: .bfloat16, hiddenSize: 4096)
    }

    func testFusedHCCollapseHandlesNoncontiguousBF16View() {
        let fixture = makeFixture()
        let source = MLXArray(
            (0..<(4 * fixture.hiddenSize)).map {
                Float(($0 % 23) - 11) / 19
            },
            [1, 1, fixture.hiddenSize, 4]).asType(.bfloat16)
        let residualView = source.transposed(0, 1, 3, 2)

        let stored = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residualView,
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let fp32 = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residualView.asType(.float32),
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        MLX.eval(stored.collapsed, fp32.collapsed)

        XCTAssertEqual(stored.collapsed.dtype, .bfloat16)
        XCTAssertTrue(
            allClose(
                stored.collapsed,
                fp32.collapsed.asType(.bfloat16),
                rtol: 0,
                atol: 0).item(Bool.self))
    }

    func testFusedHCExpandPreservesFP32ReferenceBehavior() {
        let fixture = makeFixture()
        let collapsed = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: fixture.residual,
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let block = MLXArray(
            (0..<fixture.hiddenSize).map { Float(($0 % 11) - 5) / 16 },
            [1, 1, fixture.hiddenSize])

        let fused = DeepseekV4Math.hcExpand4(
            blockOut: block,
            residual: fixture.residual,
            post: collapsed.post,
            comb: collapsed.comb,
            hiddenSize: fixture.hiddenSize)
        let reference = collapsed.post.expandedDimensions(axis: -1)
            * block.expandedDimensions(axis: -2)
            + DeepseekV4Math.hcExpandResidual(
                comb: collapsed.comb, residual: fixture.residual)
        MLX.eval(fused, reference)

        XCTAssertEqual(fused.dtype, .float32)
        XCTAssertTrue(
            allClose(fused, reference, rtol: 1e-5, atol: 1e-6)
                .item(Bool.self))
    }

    func testFusedHCStorageEligibilityFallsBackForMismatchedOrIntegerInputs() {
        let bf16 = MLXArray.zeros([1, 1, 32], dtype: .bfloat16)
        let fp32 = MLXArray.zeros([1, 1, 4, 32], dtype: .float32)
        let integer = MLXArray.zeros([1, 1, 4, 32], dtype: .int32)

        XCTAssertTrue(DeepseekV4Math.supportsFusedHCActivationStorage(.bfloat16))
        XCTAssertTrue(DeepseekV4Math.supportsFusedHCActivationStorage(.float16))
        XCTAssertTrue(DeepseekV4Math.supportsFusedHCActivationStorage(.float32))
        XCTAssertFalse(DeepseekV4Math.supportsFusedHCActivationStorage(.int32))
        XCTAssertFalse(
            DeepseekV4Math.supportsFusedHCExpansion(
                blockOut: bf16, residual: fp32))
        XCTAssertFalse(
            DeepseekV4Math.supportsFusedHCExpansion(
                blockOut: integer, residual: integer))
    }

    private func assertFusedStorageParity(dtype: DType, hiddenSize: Int) {
        let fixture = makeFixture(hiddenSize: hiddenSize)
        let residual = fixture.residual.asType(dtype)
        let collapsed = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residual,
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let collapsedFP32 = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: fixture.mixes,
            scale: fixture.scale,
            base: fixture.base,
            residual: residual.asType(.float32),
            hiddenSize: fixture.hiddenSize,
            iters: 20,
            eps: 1e-6)
        let block = MLXArray(
            (0..<fixture.hiddenSize).map { Float(($0 % 11) - 5) / 16 },
            [1, 1, fixture.hiddenSize]).asType(dtype)
        let expanded = DeepseekV4Math.hcExpand4(
            blockOut: block,
            residual: residual,
            post: collapsed.post,
            comb: collapsed.comb,
            hiddenSize: fixture.hiddenSize)
        let expandedFP32 = DeepseekV4Math.hcExpand4(
            blockOut: block.asType(.float32),
            residual: residual.asType(.float32),
            post: collapsed.post,
            comb: collapsed.comb,
            hiddenSize: fixture.hiddenSize)
        MLX.eval(
            collapsed.collapsed, collapsedFP32.collapsed,
            expanded, expandedFP32)

        XCTAssertEqual(collapsed.collapsed.dtype, dtype)
        XCTAssertEqual(expanded.dtype, dtype)
        XCTAssertTrue(
            allClose(
                collapsed.collapsed,
                collapsedFP32.collapsed.asType(dtype),
                rtol: 0,
                atol: 0).item(Bool.self))
        XCTAssertTrue(
            allClose(
                expanded,
                expandedFP32.asType(dtype),
                rtol: 0,
                atol: 0).item(Bool.self))
    }

    private func makeFixture(hiddenSize: Int = 32) -> (
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        residual: MLXArray,
        hiddenSize: Int
    ) {
        let mixes = MLXArray(
            (0..<24).map { Float(($0 % 9) - 4) / 13 },
            [1, 1, 24])
        let scale = MLXArray([Float(0.75), -0.5, 0.625])
        let base = MLXArray(
            (0..<24).map { Float(($0 % 7) - 3) / 17 })
        let residual = MLXArray(
            (0..<(4 * hiddenSize)).map { Float(($0 % 23) - 11) / 19 },
            [1, 1, 4, hiddenSize])
        return (mixes, scale, base, residual, hiddenSize)
    }
}
