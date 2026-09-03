import Foundation
import MLX

/// Model-lifetime cache for a fixed-shape compiled unary graph.
///
/// Models prepare the compiled function only after checkpoint publication and
/// invalidate it before any parameter update. The first accepted input binds
/// the dtype; later shape or dtype changes fail closed to the eager path.
/// Preparation and first-use publication are serialized, while the immutable
/// compiled function can be replayed without forward-time mutation.
public final class FixedShapeCompiledUnaryCache {
    public typealias Function = (MLXArray) -> MLXArray

    private let lock = NSLock()
    private var function: Function?
    private var expectedShape: [Int]?
    private var boundDType: DType?

    public init() {}

    public var isPrepared: Bool {
        lock.lock()
        defer { lock.unlock() }
        return function != nil
    }

    public var inputDType: DType? {
        lock.lock()
        defer { lock.unlock() }
        return boundDType
    }

    /// Publish a fresh fixed-shape graph. Calling this again replaces any
    /// earlier graph, which is useful after a model parameter update.
    public func prepare(
        enabled: Bool,
        expectedShape: [Int],
        body: @escaping Function
    ) {
        lock.lock()
        defer { lock.unlock() }
        function = nil
        self.expectedShape = nil
        boundDType = nil
        guard enabled, !expectedShape.isEmpty, expectedShape.allSatisfy({ $0 > 0 }) else {
            return
        }
        self.expectedShape = expectedShape
        function = compile(shapeless: false) { input in
            CompiledDecodeTrace.withActive { body(input) }
        }
    }

    /// Discard the graph before registered parameters change.
    public func invalidate() {
        lock.lock()
        defer { lock.unlock() }
        function = nil
        expectedShape = nil
        boundDType = nil
    }

    /// Execute only at the prepared shape and the dtype established by the
    /// first call. Returning nil asks the owning model to use its eager path.
    public func callAsFunction(_ input: MLXArray) -> MLXArray? {
        lock.lock()
        guard let function, input.shape == expectedShape else {
            lock.unlock()
            return nil
        }
        if let boundDType, boundDType != input.dtype {
            lock.unlock()
            return nil
        }
        if boundDType == nil {
            boundDType = input.dtype
        }
        lock.unlock()
        return function(input)
    }
}
