import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Text/chat preparation for a GGUF model paired with standard Hugging Face
/// tokenizer assets. This mirrors MLXLLM's private text processor so the rest
/// of AFM's generation pipeline remains unchanged.
struct AFMMLXGGUFUserInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let messageGenerator: MessageGenerator

    init(tokenizer: Tokenizer, model: any LLMModel) {
        self.tokenizer = tokenizer
        self.messageGenerator = model.messageGenerator(tokenizer: tokenizer)
    }

    func prepare(input: UserInput) throws -> LMInput {
        if case .text(let prompt) = input.prompt {
            return LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
        }

        let messages = messageGenerator.generate(from: input)
        do {
            let chatTemplate: ChatTemplateArgument?
            if let override = input.additionalContext?["chatTemplateOverride"] as? String {
                chatTemplate = .literal(override)
            } else {
                chatTemplate = nil
            }
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: chatTemplate,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(tokens))
        } catch TokenizerError.missingChatTemplate {
            let prompt = messages.compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            return LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
        }
    }
}
