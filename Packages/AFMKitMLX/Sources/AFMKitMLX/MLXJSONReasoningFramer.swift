import Foundation

/// Request-local framing for structured output. Reasoning may precede prompted
/// JSON, but marker-shaped strings inside the JSON are ordinary data. Shares
/// the same one-way transition between stop filtering and event translation.
/// No model/GPU state is retained; only a possible partial delimiter is buffered.
struct MLXJSONReasoningFramer {
    struct Segment {
        let text: String
        let isReasoning: Bool
        var isDelimiter = false
    }

    private let startTag: String?
    private let endTag: String?
    private var insideReasoning: Bool
    private var jsonStarted = false
    private var buffer = ""

    init(startTag: String?, endTag: String?, insideReasoning: Bool = false) {
        self.startTag = startTag
        self.endTag = endTag
        self.insideReasoning = insideReasoning
    }

    mutating func consume(_ text: String) -> [Segment] {
        guard !text.isEmpty else { return [] }
        guard !jsonStarted, let startTag, let endTag, !startTag.isEmpty, !endTag.isEmpty else {
            return [.init(text: text, isReasoning: false)]
        }
        buffer += text
        var segments: [Segment] = []
        while !buffer.isEmpty {
            let delimiter = insideReasoning ? endTag : startTag
            let boundary = buffer.range(of: delimiter)
            // Look before the next reasoning delimiter, not merely at the first
            // character: JSON can follow a markdown fence. A root string is
            // also JSON and must preserve literal reasoning markers.
            if !insideReasoning,
               let json = buffer.firstIndex(where: { $0 == "{" || $0 == "[" || $0 == "\"" }),
               boundary == nil || json < boundary!.lowerBound {
                jsonStarted = true
                segments.append(.init(text: buffer, isReasoning: false))
                buffer = ""
                break
            }
            if let boundary {
                if boundary.lowerBound != buffer.startIndex {
                    segments.append(.init(text: String(buffer[..<boundary.lowerBound]),
                                          isReasoning: insideReasoning))
                }
                segments.append(.init(text: delimiter, isReasoning: true, isDelimiter: true))
                buffer = String(buffer[boundary.upperBound...])
                insideReasoning.toggle()
                continue
            }
            let retained = (1..<delimiter.count).reversed().first {
                buffer.hasSuffix(String(delimiter.prefix($0)))
            } ?? 0
            let end = buffer.index(buffer.endIndex, offsetBy: -retained)
            if end != buffer.startIndex {
                segments.append(.init(text: String(buffer[..<end]), isReasoning: insideReasoning))
                buffer = String(buffer[end...])
            }
            break
        }
        return segments
    }

    mutating func finish() -> [Segment] {
        guard !buffer.isEmpty else { return [] }
        defer { buffer = "" }
        return [.init(text: buffer, isReasoning: insideReasoning)]
    }
}

/// Stops apply to visible JSON, never to leading reasoning. Delimiters are kept
/// here so downstream reasoning extraction still receives a complete frame.
struct MLXJSONStopFilter {
    private var framer: MLXJSONReasoningFramer
    private var stops: MLXStreamingStopBuffer
    private(set) var stopped = false
    private var inputBytes = 0
    private var emittedBytes = 0
    private var pendingLogprobs: [(end: Int, values: [ResolvedLogprob])] = []

    init(startTag: String?, endTag: String?, insideReasoning: Bool, stopSequences: [String]) {
        framer = .init(startTag: startTag, endTag: endTag, insideReasoning: insideReasoning)
        stops = .init(stopSequences: stopSequences)
    }

    mutating func consume(_ text: String, logprobs: [ResolvedLogprob]? = nil) -> [StreamChunk] {
        guard !stopped else { return [] }
        inputBytes += text.utf8.count
        if let logprobs, !logprobs.isEmpty {
            pendingLogprobs.append((inputBytes, logprobs))
        }
        return annotate(filter(framer.consume(text)))
    }

    mutating func finish() -> [StreamChunk] {
        guard !stopped else { return [] }
        var chunks = filter(framer.finish())
        let tail = stops.finish()
        if !tail.isEmpty { chunks.append(.init(text: tail)) }
        return annotate(chunks)
    }

    private mutating func annotate(_ chunks: [StreamChunk]) -> [StreamChunk] {
        // Output remains an exact prefix of input: reasoning delimiters are
        // retained here, and only a stop and its suffix are removed. Release
        // token metadata only once its complete originating text is visible.
        // Tokens intersecting a removed stop are conservatively suppressed.
        let result = chunks.map { chunk in
            emittedBytes += chunk.text.utf8.count
            var visible: [ResolvedLogprob] = []
            while let first = pendingLogprobs.first, first.end <= emittedBytes {
                visible += first.values
                pendingLogprobs.removeFirst()
            }
            return StreamChunk(text: chunk.text, logprobs: visible.isEmpty ? nil : visible,
                               stoppedBySequence: chunk.stoppedBySequence)
        }
        if stopped { pendingLogprobs.removeAll() }
        return result
    }

    private mutating func filter(_ segments: [MLXJSONReasoningFramer.Segment]) -> [StreamChunk] {
        var chunks: [StreamChunk] = []
        for segment in segments {
            if segment.isReasoning {
                let visible = stops.finish()
                if !visible.isEmpty { chunks.append(.init(text: visible)) }
                chunks.append(.init(text: segment.text))
            } else {
                let result = stops.consume(segment.text)
                if let text = result.text {
                    chunks.append(.init(text: text, stoppedBySequence: result.stopped ? true : nil))
                }
                if result.stopped {
                    stopped = true
                    break
                }
            }
        }
        return chunks
    }
}
