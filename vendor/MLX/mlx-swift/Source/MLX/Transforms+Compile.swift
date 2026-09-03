// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// Note: this is all immutable state -- the `id` property is only set at init time
final class CompiledFunction: @unchecked (Sendable) {

    private static let identifierLock = NSLock()
    nonisolated(unsafe) private static var nextIdentifier: UInt = 1

    private static func makeIdentifier() -> UInt {
        identifierLock.withLock {
            let result = nextIdentifier
            nextIdentifier &+= 1
            precondition(nextIdentifier != 0, "MLX compiled-function identifier exhausted")
            return result
        }
    }

    /// unique (for the lifetime of the object) identifier for the compiled function
    private let id: UInt

    /// A cache owned by this compiled function. Swift tasks may resume on a
    /// different OS thread between calls, so the default thread-local MLX cache
    /// cannot safely express the function's lifetime. Keeping the cache here
    /// also releases graphs and captured model weights when the model dies.
    private let cache: mlx_detail_compile_cache

    let lock = NSLock()

    /// the function to compile
    let f: ([MLXArray]) -> [MLXArray]

    /// any state to be observed
    let inputs: [any Updatable]
    let outputs: [any Updatable]

    let shapeless: Bool

    init(
        inputs: [any Updatable], outputs: [any Updatable], shapeless: Bool,
        _ f: @escaping ([MLXArray]) -> [MLXArray]
    ) {
        self.f = f
        self.inputs = inputs
        self.outputs = outputs
        self.shapeless = shapeless
        self.id = Self.makeIdentifier()
        self.cache = mlx_detail_compile_cache_new()
        precondition(cache.ctx != nil, "Unable to create MLX compilation cache")
    }

    deinit {
        // Releasing the owned cache removes every shape specialization and any
        // retained model arrays independent of the executor thread running deinit.
        mlx_detail_compile_cache_free(cache)
    }

    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        // Keep the global MLX evaluation lock outermost. A compiled closure may
        // trace another compiled helper. Acquiring a function lock first lets
        // two concurrent traces invert the order (function A -> eval ->
        // function B versus function B -> eval) and deadlock. Every other MLX
        // evaluation entry point already uses evalLock as the serialization
        // boundary, so use the same order here.
        evalLock.withLock {
            lock.withLock {
                innerCall(arguments)
            }
        }
    }

    func innerCall(_ arguments: [MLXArray]) -> [MLXArray] {
        let stateInputs = inputs.flatMap { $0.innerState() }
        let argumentsCount = arguments.count

        // inner function to hande the compilation.  this is called
        // once per compile (typically once overall, but can be called
        // again if the conditions for recompile change)
        func inner(tracers: [MLXArray]) -> [MLXArray] {

            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracerArguments = Array(tracers.prefix(argumentsCount))

            // save a snapshot of the inner state
            let savedStateInputs = stateInputs.map { $0.copyContext() }

            // replace the inner state with the tracers
            for (s, tracer) in zip(stateInputs, tracers[argumentsCount...]) {
                s._updateInternal(tracer)
            }

            // call the function with the tracer arguments
            // and the state holding tracers
            let result = f(tracerArguments)

            // recapture the state as it may have changed
            let stateOutputTracers = outputs.flatMap { $0.innerState() }.map { $0.copyContext() }

            // put the original values back in the state
            for (s, saved) in zip(stateInputs, savedStateInputs) {
                s._updateInternal(saved)
            }

            // return the result of the function and the state
            return result + stateOutputTracers
        }

        let innerClosure = new_mlx_closure(inner(tracers:))
        defer { mlx_closure_free(innerClosure) }

        // note: this will use the cached compile (via the id)
        // but will be able to re-evaluate with fresh state if needed
        var compiled = mlx_closure_new()
        let compileStatus = mlx_detail_compile_with_cache(
            &compiled, innerClosure, id, shapeless, [], 0, cache)
        defer {
            mlx_closure_free(compiled)
        }

        // mlx_error was already dispatched on failure:
        //   • outside withError — fatalError was called; we won't reach here
        //   • inside withError  — error is stored in the ErrorBox; return [] so
        //                         withError can throw instead of crashing downstream
        guard compileStatus == 0 else {
            return []
        }

        let innerInputs = arguments + stateInputs
        let innerInputsVector = new_mlx_vector_array(innerInputs)
        defer { mlx_vector_array_free(innerInputsVector) }

        // will compile the function (if needed) and evaluate the
        // compiled graph
        var resultVector = mlx_vector_array_new()
        let applyStatus = mlx_closure_apply(&resultVector, compiled, innerInputsVector)
        defer { mlx_vector_array_free(resultVector) }

        guard applyStatus == 0 else {
            return []
        }

        let resultsPlusStateOutput = mlx_vector_array_values(resultVector)

        // push the stateOutput into the state
        let stateOutput = outputs.flatMap { $0.innerState() }

        for (s, newValues) in zip(stateOutput, resultsPlusStateOutput.suffix(stateOutput.count)) {
            s._updateInternal(newValues)
        }

        let resultLength = resultsPlusStateOutput.count - stateOutput.count
        let results = Array(resultsPlusStateOutput.prefix(resultLength))
        return results
    }
}

/// Returns a compiled function that produces the same output as `f()`.
///
/// Any mutable state must be provided via the state parameter -- see <doc:compilation> for more
/// information.
///
/// - Parameters:
///   - inputs: input state
///   - outputs: output state
///   - shapeless: A function compiled with the `shapeless`
///     option enabled will not be recompiled when the input shape changes. Not all
///     functions can be compiled with `shapeless` enabled. Attempting to compile
///     such functions with shapeless enabled will throw. Note, changing the number
///     of dimensions or type of any input will result in a recompilation even with
///     `shapeless` set to `true`
///   - f: function to compile
/// - Returns: a new function that produces the same output as `f()`
///
/// ### See Also
/// - <doc:compilation>
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless, f)

    return { arrays in
        compileState.call(arrays)
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes a single ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> MLXArray
) -> @Sendable (MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0])]
    }

    return { a in
        let r = compileState.call([a])
        // r is empty only when an MLX error fired inside a withError scope — the
        // error is already stored in the ErrorBox.  Return a placeholder so that
        // withError can throw instead of crashing with "Index out of range".
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes two ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1])]
    }

    return { a, b in
        let r = compileState.call([a, b])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes three ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @Sendable @escaping (MLXArray, MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1], $0[2])]
    }

    return { a, b, c in
        let r = compileState.call([a, b, c])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Globally enable or disable ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``.
///
/// Default is enabled.
public func compile(enable: Bool = true) {
    if enable {
        mlx_enable_compile()
    } else {
        mlx_disable_compile()
    }
}
