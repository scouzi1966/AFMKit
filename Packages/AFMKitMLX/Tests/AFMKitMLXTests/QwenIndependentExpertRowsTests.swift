import Foundation
import MLX
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

final class QwenIndependentExpertRowsTests: XCTestCase {
    // The optional benchmark freezes all parameters before wrapping this layer.
    // Both compiled functions run serially on this test's thread; the wrapper
    // permits compile's Sendable closure without claiming SwitchGLU is mutable
    // safely across threads in production.
    private final class FrozenLayer: @unchecked Sendable {
        let value: SwitchGLU
        init(_ value: SwitchGLU) { self.value = value }
    }

    private let dimensions = 2560
    private let hiddenDimensions = 640

    private func layer(experts: Int, group: Int = 32) -> SwitchGLU {
        let result = SwitchGLU(inputDims: dimensions, hiddenDims: hiddenDimensions, numExperts: experts)
        result.update(parameters: result.mapParameters { $0.asType(.bfloat16) })
        quantize(model: result, groupSize: group, bits: 4)
        return result
    }

    private func arguments(width: Int, routes: Int, experts: Int, overlap: Bool) -> [MLXArray] {
        let input = MLXRandom.normal([1, width, dimensions]).asType(.bfloat16)
        let ids = (0..<(width * routes)).map { i in
            UInt32(((i % routes) * 7 + (overlap ? 0 : (i / routes) * routes * 13)) % experts)
        }
        let scores = softmax(MLXRandom.normal([1, width, routes]).asType(.bfloat16))
        return [input, MLXArray(ids).reshaped(1, width, routes), scores]
    }

    private func singleton(_ layer: SwitchGLU, _ a: [MLXArray]) -> MLXArray {
        concatenated((0..<a[0].dim(1)).map { row in
            layer.qwenAffineDecode(a[0][0..., row..<(row + 1), 0...],
                indices: a[1][0..., row..<(row + 1), 0...],
                scores: a[2][0..., row..<(row + 1), 0...])!
        }, axis: 1)
    }

    func testCompleteOutputsMatchIndependentSingletons() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(313)
        let model = layer(experts: 32)
        eval(model)
        model.prepareQwenAffineDecode()
        for width in [2, 4, 7, 8] {
            for routes in [1, 3, 10, 15, 31] {
                for overlap in [true, false] {
                    let a = arguments(width: width, routes: routes, experts: 32, overlap: overlap)
                    let expected = singleton(model, a)
                    let actual = try XCTUnwrap(model.qwenIndependentAffineRows(
                        a[0], indices: a[1], scores: a[2]))
                    if !arrayEqual(expected, actual).item(Bool.self) {
                        let e = expected.asType(.float32).asArray(Float.self)
                        let v = actual.asType(.float32).asArray(Float.self)
                        let differences = e.indices.filter { e[$0] != v[$0] }
                        let first = differences.prefix(6).map { ($0, e[$0], v[$0]) }
                        print("Expert mismatch width=\(width) routes=\(routes) overlap=\(overlap): "
                            + "count=\(differences.count) first=\(first)")
                        let aligned = a.map { $0 }
                        let copiedSingleton = concatenated((0..<width).map { row in
                            let x = aligned[0][0..., row..<(row + 1), 0...]
                            let ids = aligned[1][0..., row..<(row + 1), 0...]
                            let scores = aligned[2][0..., row..<(row + 1), 0...]
                            let copied = MLXArray(scores.asArray(Float.self)).asType(.bfloat16)
                                .reshaped(scores.shape)
                            return model.qwenAffineDecode(x, indices: ids, scores: copied)!
                        }, axis: 1)
                        print("Fresh aligned singleton scores match candidate: "
                            + "\(arrayEqual(copiedSingleton, actual).item(Bool.self))")
                    }
                    XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self),
                        "width=\(width) routes=\(routes) overlap=\(overlap)")
                    let order = MLXArray((0..<width).reversed().map(Int32.init))
                    let reversed = a.map { take($0, order, axis: 1) }
                    let reordered = try XCTUnwrap(model.qwenIndependentAffineRows(
                        reversed[0], indices: reversed[1], scores: reversed[2]))
                    XCTAssertTrue(arrayEqual(reordered, take(expected, order, axis: 1)).item(Bool.self))
                }
            }
        }
    }

    func testSmallProjectionAndNoncontiguousViewsMatchSingletons() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(523)
        let inputDimensions = 256
        let hiddenDimensions = 96
        let expertCount = 8
        let model = SwitchGLU(inputDims: inputDimensions, hiddenDims: hiddenDimensions,
                              numExperts: expertCount)
        model.update(parameters: model.mapParameters { $0.asType(.bfloat16) })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        model.prepareQwenAffineDecode()
        for width in [2, 4, 7, 8] {
            for routes in [1, 3, 7] {
                // Strided features and nonzero row offsets force view handling;
                // 96 input channels in the down projection also leave idle lanes.
                let storage = MLXRandom.normal([1, width + 1, inputDimensions * 2]).asType(.bfloat16)
                let x = storage[0..., 1..<(width + 1), .stride(by: 2)]
                let indices = MLXArray((0..<(width * routes)).map { UInt32(($0 * 3) % expertCount) })
                    .reshaped(1, width, routes)
                let scores = softmax(MLXRandom.normal([1, width, routes]).asType(.bfloat16))
                let expected = singleton(model, [x, indices, scores])
                let actual = try XCTUnwrap(model.qwenIndependentAffineRows(x, indices: indices, scores: scores))
                XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self), "width=\(width) routes=\(routes)")
            }
        }
    }

    func testUnsupportedShapesFailClosed() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = layer(experts: 32)
        for width in [1, 9] {
            let a = arguments(width: width, routes: 10, experts: 32, overlap: false)
            XCTAssertNil(model.qwenIndependentAffineRows(a[0], indices: a[1], scores: a[2]))
        }
        let a = arguments(width: 4, routes: 10, experts: 32, overlap: false)
        XCTAssertNil(model.qwenIndependentAffineRows(a[0].asType(.float32), indices: a[1], scores: a[2]))
        XCTAssertNil(model.qwenIndependentAffineRows(a[0], indices: a[1], scores: a[2][0..., 0..<2, 0...]))
        let batch = a.map { concatenated([$0, $0], axis: 0) }
        XCTAssertNil(model.qwenIndependentAffineRows(batch[0], indices: batch[1], scores: batch[2]))
        let group64 = layer(experts: 32, group: 64)
        XCTAssertNil(group64.qwenIndependentAffineRows(a[0], indices: a[1], scores: a[2]))
        let tooMany = arguments(width: 4, routes: 32, experts: 32, overlap: true)
        XCTAssertNil(model.qwenIndependentAffineRows(tooMany[0], indices: tooMany[1], scores: tooMany[2]))
    }

    func testUnalignedSmallScoreViewsMatchFreshStorage() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = layer(experts: 32)
        model.prepareQwenAffineDecode()
        for routes in [1, 3, 5, 7] {
            let a = arguments(width: 4, routes: routes, experts: 32, overlap: false)
            for row in 0..<4 {
                let x = a[0][0..., row..<(row + 1), 0...]
                let ids = a[1][0..., row..<(row + 1), 0...]
                let scores = a[2][0..., row..<(row + 1), 0...]
                let copied = MLXArray(scores.asArray(Float.self)).asType(.bfloat16)
                    .reshaped(scores.shape)
                let fromView = try XCTUnwrap(model.qwenAffineDecode(x, indices: ids, scores: scores))
                let fromCopy = try XCTUnwrap(model.qwenAffineDecode(x, indices: ids, scores: copied))
                XCTAssertTrue(arrayEqual(fromView, fromCopy).item(Bool.self),
                              "routes=\(routes), row=\(row)")
            }
        }
    }

    func testOptionalRealExpertLatency() throws {
        #if DEBUG
        throw XCTSkip("Release-only optional latency experiment")
        #else
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_QWEN_EXPERT_ROW_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external report path required")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let root = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "IndependentExpertProbe", code: 1)
        }
        let index = try JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let map = try XCTUnwrap(index?["weight_map"] as? [String: String])
        let model = layer(experts: 512)
        let names = model.parameters().flattened().map(\.0)
        let prefix = "language_model.model.layers.0.mlp.switch_mlp."
        let files = try Set(names.map { try XCTUnwrap(map[prefix + $0]) })
        var parameters: [(String, MLXArray)] = []
        for name in files.sorted() {
            guard name == URL(fileURLWithPath: name).lastPathComponent else {
                throw NSError(domain: "IndependentExpertProbe", code: 2)
            }
            let file = root.appendingPathComponent(name).resolvingSymlinksInPath()
            let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            guard file.deletingLastPathComponent() == root, bytes < 6 * 1024 * 1024 * 1024 else {
                throw NSError(domain: "IndependentExpertProbe", code: 3)
            }
            let tensors = try MLX.loadArrays(url: file)
            for key in names where map[prefix + key] == name {
                parameters.append((key, try XCTUnwrap(tensors[prefix + key])))
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(model)
        model.prepareQwenAffineDecode()
        MLXRandom.seed(317)
        let frozen = FrozenLayer(model)
        var samples: [[String: Any]] = []
        for width in [2, 4, 7, 8] {
            for overlap in [true, false] {
                let a = arguments(width: width, routes: 10, experts: 512, overlap: overlap)
                let baseline: @Sendable ([MLXArray]) -> [MLXArray] = { a in
                    [concatenated((0..<width).map { row in
                        frozen.value.qwenAffineDecode(a[0][0..., row..<(row + 1), 0...],
                            indices: a[1][0..., row..<(row + 1), 0...],
                            scores: a[2][0..., row..<(row + 1), 0...])!
                    }, axis: 1)]
                }
                let candidate: @Sendable ([MLXArray]) -> [MLXArray] = { a in
                    [frozen.value.qwenIndependentAffineRows(a[0], indices: a[1], scores: a[2])!]
                }
                let functions = [compile(shapeless: false, baseline), compile(shapeless: false, candidate)]
                eval(a)
                XCTAssertTrue(arrayEqual(functions[0](a)[0], functions[1](a)[0]).item(Bool.self),
                    "real weights width=\(width) overlap=\(overlap)")
                for trial in 0..<24 {
                    for i in 0..<2 {
                        let arm = (i + trial) % 2
                        Stream.gpu.synchronize()
                        let start = DispatchTime.now().uptimeNanoseconds
                        let output = functions[arm](a)
                        eval(output)
                        Stream.gpu.synchronize()
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                        if trial >= 4 {
                            samples.append(["width": width, "overlap": overlap, "arm": arm,
                                            "trial": trial, "milliseconds": ms])
                        }
                    }
                }
                let medians = (0..<2).map { arm in
                    samples.filter { ($0["width"] as? Int) == width
                        && ($0["overlap"] as? Bool) == overlap && ($0["arm"] as? Int) == arm }
                        .compactMap { $0["milliseconds"] as? Double }.sorted()[10]
                }
                print("Independent experts width=\(width) overlap=\(overlap): singleton=\(medians[0])ms candidate=\(medians[1])ms")
            }
        }
        let document: [String: Any] = ["checkpoint": root.path, "layer": 0,
            "experts": 512, "routes": 10, "synthetic_hidden_and_routes": true, "samples": samples]
        try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: [.withoutOverwriting])
        #endif
    }
}
