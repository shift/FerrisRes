use std::collections::HashMap;

pub struct SimpleTokenizer {
    vocab: Vec<String>,
    #[allow(dead_code)]
    token_to_id: HashMap<String, u32>,
    #[allow(dead_code)]
    unk_id: u32,
    eos_id: u32,
    #[allow(dead_code)]
    pad_id: u32,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut vocab = Vec::with_capacity(259);
        let mut token_to_id = HashMap::with_capacity(259);

        vocab.push("<unk>".to_string());
        token_to_id.insert("<unk>".to_string(), 0);

        vocab.push("<eos>".to_string());
        token_to_id.insert("<eos>".to_string(), 1);

        vocab.push("<pad>".to_string());
        token_to_id.insert("<pad>".to_string(), 2);

        for b in 0u32..256u32 {
            let token = format!("<0x{:02X}>", b);
            token_to_id.insert(token.clone(), b + 3);
            vocab.push(token);
        }

        tracing::info!("SimpleTokenizer created with vocab_size={}", vocab.len());

        Self {
            vocab,
            token_to_id,
            unk_id: 0,
            eos_id: 1,
            pad_id: 2,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32 + 3).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if id >= 3 && id < 259 {
                bytes.push((id - 3) as u8);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// BPE (Byte-Pair Encoding) subword tokenizer
#[allow(dead_code)]
pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    vocab_size: usize,
    unk_id: u32,
    eos_id: u32,
    bos_id: u32,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with given vocab size
    pub fn new(vocab_size: usize) -> Self {
        // Initialize with byte-level tokens
        let mut token_to_id = HashMap::new();
        let mut vocab = Vec::new();
        
        // Special tokens
        let specials = ["<unk>", "<s>", "</s>", "<pad>"];
        for (i, tok) in specials.iter().enumerate() {
            token_to_id.insert(tok.to_string(), i as u32);
            vocab.push(tok.to_string());
        }
        
        // Byte tokens
        for b in 0u8..=255u8 {
            let token = format!("{}", b as char);
            token_to_id.insert(token.clone(), vocab.len() as u32);
            vocab.push(token);
        }
        
        let unk_id = 0;
        let eos_id = 2;
        let bos_id = 1;
        
        Self {
            vocab,
            token_to_id,
            merges: Vec::new(),
            vocab_size,
            unk_id,
            eos_id,
            bos_id,
        }
    }
    
    /// Create with default vocab size (32K)
    pub fn default_vocab() -> Self {
        Self::new(32000)
    }
    
    /// Train tokenizer on text corpus (computes merges)
    pub fn train(&mut self, text: &str, max_merges: usize) {
        // Simple training: collect frequent pairs
        let mut pairs: HashMap<(u8, u8), usize> = HashMap::new();
        
        // Count byte pairs
        for word in text.split_whitespace() {
            let bytes = word.as_bytes();
            for window in bytes.windows(2) {
                *pairs.entry((window[0], window[1])).or_insert(0) += 1;
            }
        }
        
        // Sort by frequency and take top merges
        let mut sorted_pairs: Vec<_> = pairs.into_iter().collect();
        sorted_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        for ((b1, b2), _) in sorted_pairs.into_iter().take(max_merges) {
            let pair = (format!("{}", b1 as char), format!("{}", b2 as char));
            let pair_clone = pair.clone();
            self.merges.push(pair_clone);
            
            // Add to vocab
            let merged = format!("{}{}", pair.0, pair.1);
            let merged_key = merged.clone();
            if !self.token_to_id.contains_key(&merged_key) {
                let vocab_len = self.vocab.len();
                self.token_to_id.insert(merged_key, vocab_len as u32);
                self.vocab.push(merged);
            }
        }
        
        tracing::info!("BPE trained with {} merges, vocab size = {}", self.merges.len(), self.vocab.len());
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Start with byte-level encoding
        let mut current: Vec<String> = text.bytes()
            .map(|b| format!("{}", b as char))
            .collect();
        
        // Apply merges (simplified - just greedy longest match)
        for (a, b) in &self.merges {
            let mut next = Vec::new();
            let mut i = 0;
            while i < current.len() {
                if i + 1 < current.len() && &current[i] == a && &current[i + 1] == b {
                    next.push(format!("{}{}", a, b));
                    i += 2;
                } else {
                    next.push(current[i].clone());
                    i += 1;
                }
            }
            current = next;
        }
        
        // Convert to IDs
        for token in &current {
            if let Some(&id) = self.token_to_id.get(token) {
                tokens.push(id);
            } else {
                tokens.push(self.unk_id);
            }
        }
        
        tokens
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < self.vocab.len() as u32 {
                let token = &self.vocab[id as usize];
                // For byte tokens, just get the byte value
                if token.len() == 1 {
                    if let Some(b) = token.as_bytes().first() {
                        bytes.push(*b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::default_vocab()
    }
}

/// Domain-specific vocabulary extension for specialized tokens
/// Allows adding domain-specific tokens (e.g., SMILES, genomic sequences)
pub struct DomainVocabulary {
    base_tokenizer: BpeTokenizer,
    domain_tokens: HashMap<String, u32>,
    domain_offset: u32,
}

impl DomainVocabulary {
    /// Create new domain vocabulary extending a base BPE tokenizer
    pub fn new(base: BpeTokenizer) -> Self {
        Self {
            base_tokenizer: base,
            domain_tokens: HashMap::new(),
            domain_offset: 0, // Will be set to base vocab size
        }
    }
    
    /// Add domain-specific tokens (e.g., SMILES, gene sequences)
    pub fn extend(&mut self, tokens: Vec<String>) {
        if self.domain_offset == 0 {
            self.domain_offset = self.base_tokenizer.vocab_size() as u32;
        }
        
        for token in tokens {
            if !self.domain_tokens.contains_key(&token) {
                let id = self.domain_offset + self.domain_tokens.len() as u32;
                self.domain_tokens.insert(token, id);
            }
        }
        
        tracing::info!("Domain vocabulary extended with {} tokens", self.domain_tokens.len());
    }
    
    /// Encode text with domain-specific token handling
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // First try to match domain tokens
        let mut tokens = Vec::new();
        let mut remaining = text;
        
        while !remaining.is_empty() {
            // Try longest domain token match first
            let mut matched = false;
            for (token, id) in &self.domain_tokens {
                if remaining.starts_with(token) {
                    tokens.push(*id);
                    remaining = &remaining[token.len()..];
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                // Fall back to base tokenizer
                let base_ids = self.base_tokenizer.encode(remaining);
                tokens.extend(base_ids);
                break;
            }
        }
        
        tokens
    }
    
    /// Decode with domain-specific token handling
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        
        for &id in ids {
            if id >= self.domain_offset {
                // Domain token
                let domain_id = id - self.domain_offset;
                for (token, &tid) in &self.domain_tokens {
                    if tid == domain_id {
                        result.push_str(token);
                        break;
                    }
                }
            } else {
                // Base token - just append placeholder
                result.push('?');
            }
        }
        
        result
    }
    
    /// Get total vocabulary size (base + domain)
    pub fn total_vocab_size(&self) -> usize {
        self.base_tokenizer.vocab_size() + self.domain_tokens.len()
    }
}
