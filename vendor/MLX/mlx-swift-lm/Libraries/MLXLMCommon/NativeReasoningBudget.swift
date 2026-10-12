// Copyright © 2026 AFMKit contributors.

/// An opt-in, native-token reasoning boundary. This is not request/API policy:
/// the caller supplies checkpoint-qualified atomic markers and encoded suffix.
/// Nothing maps effort words, changes templates, or enables this by default.
/// Policy attribution: mlx-serve v26.10.1, commit
/// 02bee553f48cd3bc7d82aba0f8073820bd924738: src/generate.zig:750
/// (ThinkBound), src/scheduler.zig:7405 (one-shot room check), and
/// src/server.zig:1198 (Qwen early-stop sentence plus closer/separator).
/// That server's medium-effort mapping is not implemented by this primitive.
/// Terminal IDs are supplied by the caller; this is not an EOS, padding-loop,
/// grammar, or arbitrary stop-sequence detector.
public struct NativeReasoningBudget: Sendable {
    public enum ConfigurationError: Error, Equatable {
        case invalidConfiguration
        case unsupportedIteratorConfiguration
    }

    public let tokenLimit: Int
    public let openerTokenID: Int?
    public let closerTokenID: Int
    public let forcedTokenIDs: [Int]
    public let startsInsideReasoning: Bool
    public let terminalTokenIDs: Set<Int>

    public init(
        tokenLimit: Int, openerTokenID: Int?, closerTokenID: Int,
        forcedTokenIDs: [Int], startsInsideReasoning: Bool,
        terminalTokenIDs: Set<Int>
    ) throws {
        func validID(_ id: Int) -> Bool { id >= 0 && id <= Int(Int32.max) }
        guard tokenLimit >= 0, validID(closerTokenID),
            openerTokenID.map(validID) ?? true,
            openerTokenID != closerTokenID,
            !terminalTokenIDs.isEmpty, terminalTokenIDs.allSatisfy(validID),
            !terminalTokenIDs.contains(closerTokenID),
            openerTokenID.map({ !terminalTokenIDs.contains($0) }) ?? true,
            (1...64).contains(forcedTokenIDs.count), forcedTokenIDs.allSatisfy(validID),
            forcedTokenIDs.filter({ $0 == closerTokenID }).count == 1,
            !forcedTokenIDs.contains(where: terminalTokenIDs.contains),
            openerTokenID.map({ !forcedTokenIDs.contains($0) }) ?? true
        else { throw ConfigurationError.invalidConfiguration }
        self.tokenLimit = tokenLimit
        self.openerTokenID = openerTokenID
        self.closerTokenID = closerTokenID
        self.forcedTokenIDs = forcedTokenIDs
        self.startsInsideReasoning = startsInsideReasoning
        self.terminalTokenIDs = terminalTokenIDs
    }
}

/// Pure host policy, independent of MLX, detokenization, and surfaced text.
/// Observe only returned native generation IDs, never prompt/history IDs.
public struct NativeReasoningBudgetState: Sendable {
    public enum Decision: Equatable, Sendable {
        case notDue
        case noRoom
        case pendingTerminal
        case force
    }

    public let configuration: NativeReasoningBudget
    public private(set) var insideReasoning: Bool
    public private(set) var reasoningTokenCount = 0
    public private(set) var fired = false

    public init(_ configuration: NativeReasoningBudget) {
        self.configuration = configuration
        self.insideReasoning = configuration.startsInsideReasoning
    }

    public mutating func observe(_ tokenID: Int) {
        if tokenID == configuration.closerTokenID {
            insideReasoning = false
        } else if tokenID == configuration.openerTokenID {
            insideReasoning = true
            reasoningTokenCount = 0
        } else if insideReasoning && reasoningTokenCount < Int.max {
            reasoningTokenCount += 1
        }
    }

    public var isDue: Bool {
        !fired && insideReasoning && reasoningTokenCount >= configuration.tokenLimit
    }

    /// The reference reserves the already-sampled pending token, the complete
    /// suffix, AND one answer token. No output allowance is created/reserved
    /// before the limit is reached. Once refused, the attempt is not retried.
    public mutating func takeDecision(
        returnedTokenCount: Int, maximumTokens: Int?, pendingTokenID: Int
    ) -> Decision {
        guard isDue else { return .notDue }
        fired = true
        if let maximumTokens {
            guard returnedTokenCount >= 0, maximumTokens >= returnedTokenCount,
                maximumTokens - returnedTokenCount >= configuration.forcedTokenIDs.count + 2
            else { return .noRoom }
        }
        if configuration.terminalTokenIDs.contains(pendingTokenID) { return .pendingTerminal }
        return .force
    }
}

/// Provenance of the most recently returned token in the opt-in iterator.
/// Forced tokens are not samples and must never receive fabricated logprobs.
public enum NativeReasoningTokenOrigin: Equatable, Sendable {
    case sampled
    case forcedBoundary
}

/// Opt-in host observations, after the iterator has already returned a native
/// ID. No tensor reads, detokenization, model work, or API accounting are added.
/// A returned ID is not necessarily visible: EOS, cancellation and buffered
/// tool text retain their ordinary streaming semantics.
public struct NativeGenerationDiagnosticObserver: Sendable {
    // Classification at observation, before the existing cancellation/stop
    // checks. `ordinary` does not assert submission or visibility.
    public enum Delivery: String, Sendable { case ordinary, eos, unknown, cancelled }
    public enum End: String, Sendable { case iteratorEnded, eos, unknown, cancelled, toolCall, consumerTerminated }
    public struct Token: Sendable {
        public let index: Int
        public let tokenID: Int
        public let origin: NativeReasoningTokenOrigin
        public let delivery: Delivery
        public let committedUndelivered: Int
        public init(index: Int, tokenID: Int, origin: NativeReasoningTokenOrigin,
            delivery: Delivery, committedUndelivered: Int) {
            self.index = index; self.tokenID = tokenID; self.origin = origin
            self.delivery = delivery; self.committedUndelivered = committedUndelivered
        }
    }
    public struct Completion: Sendable {
        public let returnedTokenCount: Int
        public let end: End
        public let abandonedTokenCount: Int
        public let cacheAtDeliveredBoundary: Bool
        public init(returnedTokenCount: Int, end: End, abandonedTokenCount: Int,
            cacheAtDeliveredBoundary: Bool) {
            self.returnedTokenCount = returnedTokenCount; self.end = end
            self.abandonedTokenCount = abandonedTokenCount
            self.cacheAtDeliveredBoundary = cacheAtDeliveredBoundary
        }
    }
    public let onToken: @Sendable (Token) -> Void
    public let onCompletion: @Sendable (Completion) -> Void

    public init(onToken: @escaping @Sendable (Token) -> Void,
        onCompletion: @escaping @Sendable (Completion) -> Void) {
        self.onToken = onToken
        self.onCompletion = onCompletion
    }
}
