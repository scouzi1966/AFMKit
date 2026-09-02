import Foundation
import MLX

/// Model-lifetime storage for a stateless fixed-shape compiled unary graph.
///
/// The owning model prepares this only after publishing checkpoint parameters
/// and invalidates it before any later parameter update. Mutable request state
/// must be passed as input instead of captured by the compiled closure.
public final class FixedShapeCompiledUnaryCache {
    public typealias Function = (MLXArray) -> MLXArray

    private let lock = NSLock()
    private var function: Function?
    private var expectedShape: [Int]?
    private var boundDType: DType?

    public init() {}

    public var isPrepared: Bool {
        lock.withLock { function != nil }
    }

    public var inputDType: DType? {
        lock.withLock { boundDType }
    }

    public func prepare(
        enabled: Bool,
        expectedShape: [Int],
        body: @escaping Function
    ) {
        lock.withLock {
            function = nil
            self.expectedShape = nil
            boundDType = nil
            guard enabled,
                  !expectedShape.isEmpty,
                  expectedShape.allSatisfy({ $0 > 0 }) else { return }
            self.expectedShape = expectedShape
            function = compile(shapeless: false) { input in
                CompiledDecodeTrace.withActive { body(input) }
            }
        }
    }

    public func invalidate() {
        lock.withLock {
            function = nil
            expectedShape = nil
            boundDType = nil
        }
    }

    /// Replays only at the prepared shape and the dtype bound by first use.
    /// Unsupported inputs return nil so the caller can preserve its eager path.
    public func callAsFunction(_ input: MLXArray) -> MLXArray? {
        let snapshot: Function? = lock.withLock {
            guard let function, input.shape == expectedShape else { return nil }
            if let boundDType, boundDType != input.dtype { return nil }
            if boundDType == nil { boundDType = input.dtype }
            return function
        }
        return snapshot?(input)
    }
}
