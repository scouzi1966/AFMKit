// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private var state = State.normal
    private var toolCallBuffer = ""
    private var activeEndTag: String?

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    // MARK: - State Enum

    private enum State {
        case normal
        case collectingToolCall
    }

    private struct TagPair {
        let start: String
        let end: String
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start/end tags).
    private var isInlineFormat: Bool {
        tagPairs.isEmpty
    }

    private var tagPairs: [TagPair] {
        var result: [TagPair] = []
        if let start = parser.startTag, let end = parser.endTag {
            result.append(TagPair(start: start, end: end))
        }
        if let start = parser.alternateStartTag, let end = parser.alternateEndTag {
            result.append(TagPair(start: start, end: end))
        }
        return result
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk)
    }

    /// Remove completed calls in generation order.
    ///
    /// `generateTask` uses this method directly, so tests can exercise the
    /// production FIFO and `stopAfterToolCall` queue semantics without a model.
    public func drainToolCalls(stopAfterFirst: Bool = false) -> [ToolCall] {
        let count = stopAfterFirst ? min(1, toolCalls.count) : toolCalls.count
        guard count > 0 else { return [] }

        let drained = Array(toolCalls.prefix(count))
        toolCalls.removeFirst(count)
        return drained
    }

    /// Return raw tagged content that remained incomplete when generation ended.
    ///
    /// Tagged parsers must not silently consume a partial call. AFM's provider
    /// fallback owns incomplete-call salvage, so normal EOS/token-limit
    /// completion forwards this text downstream. Inline parsers already return
    /// every chunk as passthrough while incomplete and therefore have nothing
    /// additional to flush.
    public func finishPendingText() -> String? {
        guard !isInlineFormat else {
            toolCallBuffer = ""
            return nil
        }

        let pending = toolCallBuffer
        state = .normal
        activeEndTag = nil
        toolCallBuffer = ""
        return pending.isEmpty ? nil : pending
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    private func processInlineChunk(_ chunk: String) -> String? {
        toolCallBuffer += chunk

        if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
            toolCalls.append(toolCall)
            toolCallBuffer = ""
            return nil
        }

        // Return chunk as-is; caller handles incomplete inline tool calls
        return chunk
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            return beginTaggedCall(in: chunk)
        case .collectingToolCall:
            toolCallBuffer += chunk
            return finishTaggedCallIfComplete()
        }
    }

    private func beginTaggedCall(in chunk: String) -> String? {
        let input = toolCallBuffer + chunk
        toolCallBuffer = ""

        let matches: [(pair: TagPair, range: Range<String.Index>)] = tagPairs.compactMap { pair in
            input.range(of: pair.start).map { (pair, $0) }
        }

        if let match = matches.min(by: { $0.range.lowerBound < $1.range.lowerBound }) {
            let leadingText = String(input[..<match.range.lowerBound])
            toolCallBuffer = String(input[match.range.lowerBound...])
            activeEndTag = match.pair.end
            state = .collectingToolCall

            let completedText = finishTaggedCallIfComplete() ?? ""
            let output = leadingText + completedText
            return output.isEmpty ? nil : output
        }

        let partialLength = longestStartTagPrefixSuffixLength(in: input)
        if partialLength > 0 {
            toolCallBuffer = String(input.suffix(partialLength))
            let output = String(input.dropLast(partialLength))
            return output.isEmpty ? nil : output
        }

        return input.isEmpty ? nil : input
    }

    private func finishTaggedCallIfComplete() -> String? {
        guard let endTag = activeEndTag,
            let endRange = toolCallBuffer.range(of: endTag)
        else { return nil }

        let captured = String(toolCallBuffer[..<endRange.upperBound])
        let trailingText = String(toolCallBuffer[endRange.upperBound...])
        let parsedCall = parser.parse(content: captured, tools: tools)

        state = .normal
        activeEndTag = nil
        toolCallBuffer = ""

        var output = ""
        if let parsedCall {
            toolCalls.append(parsedCall)
        } else {
            // A strict parse failure must remain visible to AFM's raw fallback,
            // which owns compatibility repair and coercion.
            output = captured
        }

        if !trailingText.isEmpty, let trailingOutput = processTaggedChunk(trailingText) {
            output += trailingOutput
        }
        return output.isEmpty ? nil : output
    }

    private func longestStartTagPrefixSuffixLength(in input: String) -> Int {
        var best = 0
        for pair in tagPairs {
            let maximum = min(input.count, max(0, pair.start.count - 1))
            guard maximum > best else { continue }

            for length in stride(from: maximum, through: best + 1, by: -1) {
                if pair.start.hasPrefix(String(input.suffix(length))) {
                    best = length
                    break
                }
            }
        }
        return best
    }
}
