//! Prompt template registry for chat/instruction model formats.
//!
//! Supports multiple chat formats:
//! - ChatML (used by many models)
//! - Llama 2 chat format
//! - Mistral/Mixtral chat format
//! - Alpaca format
//! - Raw (no formatting)
//!
//! Based on research task 76a78f2a.

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFormat {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>`
    ChatML,
    /// Llama 2: `[INST] <<SYS>>\nsystem\n<</SYS>>\nuser [/INST] assistant </s>`
    Llama2,
    /// Mistral: `[INST] user [/INST] assistant</s>`
    Mistral,
    /// Alpaca: `### Instruction:\nuser\n### Response:\nassistant`
    Alpaca,
    /// No formatting, just concatenate.
    Raw,
}

impl TemplateFormat {
    /// All available formats.
    pub fn all() -> &'static [TemplateFormat] {
        &[
            TemplateFormat::ChatML,
            TemplateFormat::Llama2,
            TemplateFormat::Mistral,
            TemplateFormat::Alpaca,
            TemplateFormat::Raw,
        ]
    }

    /// Parse from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "chatml" => Some(TemplateFormat::ChatML),
            "llama2" | "llama-2" => Some(TemplateFormat::Llama2),
            "mistral" => Some(TemplateFormat::Mistral),
            "alpaca" => Some(TemplateFormat::Alpaca),
            "raw" => Some(TemplateFormat::Raw),
            _ => None,
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            TemplateFormat::ChatML => "chatml",
            TemplateFormat::Llama2 => "llama2",
            TemplateFormat::Mistral => "mistral",
            TemplateFormat::Alpaca => "alpaca",
            TemplateFormat::Raw => "raw",
        }
    }
}

/// Prompt template registry: applies chat formatting to messages.
pub struct PromptTemplateRegistry {
    format: TemplateFormat,
    /// System prompt override (if set, replaces first system message).
    system_prompt: Option<String>,
    /// Whether to add generation prompt at the end.
    add_generation_prompt: bool,
}

impl PromptTemplateRegistry {
    /// Create a registry with the specified format.
    pub fn new(format: TemplateFormat) -> Self {
        Self {
            format,
            system_prompt: None,
            add_generation_prompt: true,
        }
    }

    /// Create with ChatML format.
    pub fn chatml() -> Self {
        Self::new(TemplateFormat::ChatML)
    }

    /// Create with Llama 2 format.
    pub fn llama2() -> Self {
        Self::new(TemplateFormat::Llama2)
    }

    /// Create with Mistral format.
    pub fn mistral() -> Self {
        Self::new(TemplateFormat::Mistral)
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set whether to add generation prompt.
    pub fn with_generation_prompt(mut self, add: bool) -> Self {
        self.add_generation_prompt = add;
        self
    }

    /// Apply the template to a list of messages.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self.format {
            TemplateFormat::ChatML => self.apply_chatml(messages),
            TemplateFormat::Llama2 => self.apply_llama2(messages),
            TemplateFormat::Mistral => self.apply_mistral(messages),
            TemplateFormat::Alpaca => self.apply_alpaca(messages),
            TemplateFormat::Raw => self.apply_raw(messages),
        }
    }

    /// Apply template to a single user message (convenience).
    pub fn apply_single(&self, user_message: &str) -> String {
        let messages = vec![ChatMessage::user(user_message)];
        self.apply(&messages)
    }

    /// Apply template with a system message and user message (convenience).
    pub fn apply_with_system(&self, system: &str, user: &str) -> String {
        let messages = vec![
            ChatMessage::system(system),
            ChatMessage::user(user),
        ];
        self.apply(&messages)
    }

    fn get_system(&self, messages: &[ChatMessage]) -> String {
        if let Some(ref sys) = self.system_prompt {
            sys.clone()
        } else {
            messages.iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.content.clone())
                .unwrap_or_default()
        }
    }

    fn apply_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();

        // If there's a system prompt override, emit it first
        if self.system_prompt.is_some() {
            if let Some(ref sys) = self.system_prompt {
                if !sys.is_empty() {
                    out.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
                }
            }
        }

        for msg in messages {
            // Skip system messages if we have an override (already handled above)
            if msg.role == Role::System && self.system_prompt.is_some() {
                continue;
            }
            out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role.as_str(), msg.content));
        }

        if self.add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }

        out
    }

    fn apply_llama2(&self, messages: &[ChatMessage]) -> String {
        let system = self.get_system(messages);
        let mut out = String::new();
        let mut first = true;

        for msg in messages {
            if msg.role == Role::System {
                continue; // Handled separately
            }

            match msg.role {
                Role::User => {
                    if first {
                        out.push_str("<s>[INST] ");
                        if !system.is_empty() {
                            out.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", system));
                        }
                        first = false;
                    } else {
                        out.push_str("[INST] ");
                    }
                    out.push_str(&msg.content);
                    out.push_str(" [/INST] ");
                }
                Role::Assistant => {
                    out.push_str(&msg.content);
                    out.push_str(" </s>");
                }
                Role::System => unreachable!(),
            }
        }

        out
    }

    fn apply_mistral(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        let mut first = true;

        for msg in messages {
            if msg.role == Role::System {
                continue; // Mistral doesn't support system messages natively
            }

            match msg.role {
                Role::User => {
                    if first {
                        out.push_str("<s>");
                        first = false;
                    }
                    out.push_str(&format!("[INST] {} [/INST]", msg.content));
                }
                Role::Assistant => {
                    out.push_str(&format!("{}{}", msg.content, "</s>"));
                }
                Role::System => unreachable!(),
            }
        }

        out
    }

    fn apply_alpaca(&self, messages: &[ChatMessage]) -> String {
        let system = self.get_system(messages);
        let mut out = String::new();

        if !system.is_empty() {
            out.push_str(&system);
            out.push('\n');
        }

        for msg in messages {
            if msg.role == Role::System {
                continue;
            }
            match msg.role {
                Role::User => {
                    out.push_str("### Instruction:\n");
                    out.push_str(&msg.content);
                    out.push('\n');
                }
                Role::Assistant => {
                    out.push_str("### Response:\n");
                    out.push_str(&msg.content);
                    out.push('\n');
                }
                Role::System => unreachable!(),
            }
        }

        if self.add_generation_prompt {
            out.push_str("### Response:\n");
        }

        out
    }

    fn apply_raw(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str(&msg.content);
            out.push('\n');
        }
        out
    }

    /// Get the format.
    pub fn format(&self) -> TemplateFormat {
        self.format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_basic() {
        let registry = PromptTemplateRegistry::chatml();
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi there!"),
        ];
        let result = registry.apply(&messages);

        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant"));
        assert!(result.contains("Hi there!"));
        assert!(result.contains("<|im_end|>"));
    }

    #[test]
    fn test_chatml_generation_prompt() {
        let registry = PromptTemplateRegistry::chatml();
        let result = registry.apply_single("Hello!");
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_no_generation_prompt() {
        let registry = PromptTemplateRegistry::chatml().with_generation_prompt(false);
        let result = registry.apply_single("Hello!");
        assert!(!result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama2_format() {
        let registry = PromptTemplateRegistry::llama2();
        let messages = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
        ];
        let result = registry.apply(&messages);

        assert!(result.contains("<s>"));
        assert!(result.contains("[INST]"));
        assert!(result.contains("<<SYS>>"));
        assert!(result.contains("Be helpful."));
        assert!(result.contains("<</SYS>>"));
        assert!(result.contains("What is 2+2?"));
        assert!(result.contains("[/INST]"));
        assert!(result.contains("4"));
        assert!(result.contains("</s>"));
    }

    #[test]
    fn test_mistral_format() {
        let registry = PromptTemplateRegistry::mistral();
        let messages = vec![
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi!"),
        ];
        let result = registry.apply(&messages);

        assert!(result.starts_with("<s>"));
        assert!(result.contains("[INST] Hello! [/INST]"));
        assert!(result.contains("Hi!</s>"));
    }

    #[test]
    fn test_alpaca_format() {
        let registry = PromptTemplateRegistry::new(TemplateFormat::Alpaca);
        let messages = vec![
            ChatMessage::user("What is Rust?"),
        ];
        let result = registry.apply(&messages);

        assert!(result.contains("### Instruction:"));
        assert!(result.contains("What is Rust?"));
        assert!(result.contains("### Response:"));
    }

    #[test]
    fn test_raw_format() {
        let registry = PromptTemplateRegistry::new(TemplateFormat::Raw);
        let messages = vec![
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi!"),
        ];
        let result = registry.apply(&messages);
        assert!(result.contains("Hello!"));
        assert!(result.contains("Hi!"));
        assert!(!result.contains("###"));
    }

    #[test]
    fn test_system_prompt_override() {
        let registry = PromptTemplateRegistry::chatml()
            .with_system_prompt("Custom system prompt");
        let messages = vec![
            ChatMessage::system("Original system"),
            ChatMessage::user("Hello!"),
        ];
        let result = registry.apply(&messages);
        assert!(result.contains("Custom system prompt"));
    }

    #[test]
    fn test_format_from_name() {
        assert_eq!(TemplateFormat::from_name("chatml"), Some(TemplateFormat::ChatML));
        assert_eq!(TemplateFormat::from_name("llama2"), Some(TemplateFormat::Llama2));
        assert_eq!(TemplateFormat::from_name("mistral"), Some(TemplateFormat::Mistral));
        assert_eq!(TemplateFormat::from_name("alpaca"), Some(TemplateFormat::Alpaca));
        assert_eq!(TemplateFormat::from_name("unknown"), None);
    }

    #[test]
    fn test_apply_with_system() {
        let registry = PromptTemplateRegistry::chatml();
        let result = registry.apply_with_system("Be helpful.", "Hello!");
        assert!(result.contains("Be helpful."));
        assert!(result.contains("Hello!"));
    }
}
