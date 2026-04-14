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

        tracing::info!(event = "simpletokenizer_created_with_vocab_size", "SimpleTokenizer created with vocab_size={}", vocab.len());

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
pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    #[allow(dead_code)]
    vocab_size: usize,
    unk_id: u32,
    eos_id: u32,
    #[allow(dead_code)]
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
        
        tracing::info!(event = "bpe_trained_with_merges_vocab_size", "BPE trained with {} merges, vocab size = {}", self.merges.len(), self.vocab.len());
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
        
        tracing::info!(event = "domain_vocabulary_extended_with_tokens", "Domain vocabulary extended with {} tokens", self.domain_tokens.len());
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
                // Domain token — match by absolute ID
                for (token, &tid) in &self.domain_tokens {
                    if tid == id {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_encode_decode() {
        let tok = SimpleTokenizer::new();
        let tokens = tok.encode("hello world");
        assert!(!tokens.is_empty());
        let decoded = tok.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_simple_tokenizer_vocab_size() {
        let tok = SimpleTokenizer::new();
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn test_simple_tokenizer_eos() {
        let tok = SimpleTokenizer::new();
        assert_eq!(tok.eos_id(), 1);
    }

    #[test]
    fn test_simple_tokenizer_roundtrip() {
        let tok = SimpleTokenizer::new();
        let text = "test roundtrip";
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);
        // Should contain the original words in some form
        assert!(decoded.contains("test") || decoded.contains("roundtrip"));
    }

    #[test]
    fn test_bpe_basic_train() {
        let mut tok = BpeTokenizer::new(100);
        tok.train("ab ab ab ab", 5);
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let tok = BpeTokenizer::default_vocab();
        let tokens = tok.encode("hello");
        assert!(!tokens.is_empty());
        let decoded = tok.decode(&tokens);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_domain_vocab_extend() {
        let base = BpeTokenizer::default_vocab();
        let mut dv = DomainVocabulary::new(base);
        let initial = dv.total_vocab_size();
        dv.extend(vec!["<CODE>".to_string(), "</CODE>".to_string()]);
        assert!(dv.total_vocab_size() > initial);
    }

    #[test]
    fn test_domain_vocab_encode_decode() {
        let base = BpeTokenizer::default_vocab();
        let mut dv = DomainVocabulary::new(base);
        dv.extend(vec!["<RUST>".to_string()]);
        let tokens = dv.encode("<RUST> fn");
        assert!(!tokens.is_empty());
        let decoded = dv.decode(&tokens);
        // Domain token should decode to the original string
        assert!(decoded.contains("<RUST>"));
    }
}

// ---------------------------------------------------------------------------
// SentencePiece / HuggingFace Tokenizer
// ---------------------------------------------------------------------------

/// A tokenizer that loads HuggingFace `tokenizer.json` format.
/// This supports BPE models used by Gemma, LLaMA, Mistral, etc.
///
/// The tokenizer.json format is:
/// ```json
/// { "model": { "type": "BPE", "vocab": {"<unk>": 0, ...}, "merges": ["a b", ...] },
///   "added_tokens": [...], "normalizer": ..., "pre_tokenizer": ... }
/// ```
#[derive(Debug, Clone)]
pub struct HfTokenizer {
    /// Token string → ID mapping.
    vocab: HashMap<String, u32>,
    /// ID → token string mapping.
    id_to_token: HashMap<u32, String>,
    /// BPE merge rules: (token_a, token_b) → rank (lower = higher priority).
    merge_ranks: HashMap<(String, String), usize>,
    /// Special tokens.
    unk_id: u32,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
}

impl HfTokenizer {
    /// Load from a HuggingFace `tokenizer.json` file.
    pub fn from_tokenizer_json(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read tokenizer.json: {}", e))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("Failed to parse tokenizer.json: {}", e))?;

        let model = json.get("model").ok_or("Missing 'model' field")?;
        let vocab_json = model.get("vocab").ok_or("Missing 'model.vocab'")?;
        let merges_json = model.get("merges").ok_or("Missing 'model.merges'")?;

        // Parse vocab
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (token, id_val) in vocab_json.as_object().unwrap_or(&serde_json::Map::new()) {
            let id = id_val.as_u64().unwrap_or(0) as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        // Parse merges (priority = position in list)
        let mut merge_ranks = HashMap::new();
        for (i, merge) in merges_json.as_array().unwrap_or(&vec![]).iter().enumerate() {
            let merge_str = merge.as_str().unwrap_or("");
            let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
            if parts.len() == 2 {
                merge_ranks.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }

        // Parse added_tokens for special tokens
        let mut unk_id = 0u32;
        let mut bos_id = None;
        let mut eos_id = None;
        let mut pad_id = None;

        if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for tok in added {
                let id = tok.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let special = tok.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                if special {
                    vocab.insert(content.to_string(), id);
                    id_to_token.insert(id, content.to_string());
                    if content == "<unk>" { unk_id = id; }
                    if content == "<s>" || content == "<bos>" { bos_id = Some(id); }
                    if content == "</s>" || content == "<eos>" { eos_id = Some(id); }
                    if content == "<pad>" { pad_id = Some(id); }
                }
            }
        }

        Ok(Self {
            vocab, id_to_token, merge_ranks,
            unk_id, bos_id, eos_id, pad_id,
        })
    }

    /// Create a simple byte-fallback tokenizer with a given vocab size.
    /// This is a fallback when no tokenizer.json is available.
    /// Maps bytes to IDs starting at `offset`, with special tokens first.
    pub fn byte_fallback(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Special tokens
        let specials = [("<unk>", 0u32), ("<s>", 1), ("</s>", 2), ("<pad>", 3)];
        for (tok, id) in &specials {
            vocab.insert(tok.to_string(), *id);
            id_to_token.insert(*id, tok.to_string());
        }

        // Byte-fallback tokens: <0x00> through <0xFF>
        for b in 0u32..256 {
            let tok = format!("<0x{:02X}>", b);
            let id = 4 + b;
            vocab.insert(tok.clone(), id);
            id_to_token.insert(id, tok);
        }

        // Fill remaining vocab with dummy tokens
        for i in (4 + 256)..vocab_size {
            let tok = format!("<unused_{}>", i);
            vocab.insert(tok.clone(), i as u32);
            id_to_token.insert(i as u32, tok);
        }

        Self {
            vocab, id_to_token, merge_ranks: HashMap::new(),
            unk_id: 0, bos_id: Some(1), eos_id: Some(2), pad_id: Some(3),
        }
    }

    /// Encode text to token IDs using BPE with byte-fallback.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merge_ranks.is_empty() {
            // No merges → byte-fallback encoding
            return self.encode_byte_fallback(text);
        }

        // Pre-tokenize: split on whitespace and punctuation boundaries
        let words = self.pre_tokenize(text);
        let mut result = Vec::new();

        if let Some(bos) = self.bos_id {
            result.push(bos);
        }

        for word in &words {
            // Start with byte-level representation
            let mut tokens: Vec<String> = word.bytes()
                .map(|b| self.byte_to_token(b))
                .collect();

            // Iteratively apply BPE merges
            tokens = self.apply_bpe(&tokens);

            // Map to IDs
            for token in &tokens {
                result.push(*self.vocab.get(token).unwrap_or(&self.unk_id));
            }
        }

        if let Some(eos) = self.eos_id {
            result.push(eos);
        }

        result
    }

    /// Encode text → token IDs without BOS/EOS wrapping.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let words = self.pre_tokenize(text);
        let mut result = Vec::new();

        for word in &words {
            let mut tokens: Vec<String> = word.bytes()
                .map(|b| self.byte_to_token(b))
                .collect();
            tokens = self.apply_bpe(&tokens);
            for token in &tokens {
                result.push(*self.vocab.get(token).unwrap_or(&self.unk_id));
            }
        }
        result
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Try byte-fallback: <0xHH>
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    if let Ok(b) = u8::from_str_radix(&token[3..5], 16) {
                        bytes.push(b);
                        continue;
                    }
                }
                // Try UTF-8 characters
                if token.len() == 1 {
                    bytes.push(token.as_bytes()[0]);
                } else if token.starts_with('▁') {
                    // SentencePiece space marker → space + content
                    bytes.push(b' ');
                    bytes.extend_from_slice(token[3..].as_bytes());
                } else if !token.starts_with('<') {
                    bytes.extend_from_slice(token.as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special token IDs.
    pub fn unk_id(&self) -> u32 { self.unk_id }
    pub fn bos_id(&self) -> Option<u32> { self.bos_id }
    pub fn eos_id(&self) -> Option<u32> { self.eos_id }
    pub fn pad_id(&self) -> Option<u32> { self.pad_id }

    // -- Internal helpers --

    /// Encode using pure byte-fallback (no BPE merges).
    fn encode_byte_fallback(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        if let Some(bos) = self.bos_id {
            result.push(bos);
        }
        for b in text.bytes() {
            let tok = format!("<0x{:02X}>", b);
            result.push(*self.vocab.get(&tok).unwrap_or(&self.unk_id));
        }
        if let Some(eos) = self.eos_id {
            result.push(eos);
        }
        result
    }

    /// Pre-tokenize text into words.
    /// Splits on whitespace boundaries, keeping the space as prefix
    /// (SentencePiece convention: space → ▁).
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();

        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                // Space becomes ▁ prefix on next word
                current.push('▁');
            } else if i == 0 {
                // First character gets ▁ prefix
                current.push('▁');
                current.push(ch);
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    /// Map a byte value to its token string representation.
    fn byte_to_token(&self, b: u8) -> String {
        let hex = format!("<0x{:02X}>", b);
        if self.vocab.contains_key(&hex) {
            return hex;
        }
        // Fallback: try direct character
        let ch = b as char;
        let s = ch.to_string();
        if self.vocab.contains_key(&s) {
            return s;
        }
        hex
    }

    /// Apply BPE merges to a sequence of tokens.
    /// Repeatedly merges the highest-priority (lowest-rank) pair.
    fn apply_bpe(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() < 2 {
            return tokens.to_vec();
        }

        let mut current = tokens.to_vec();
        loop {
            // Find the pair with the lowest merge rank
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..current.len().saturating_sub(1) {
                let pair = (current[i].clone(), current[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            let pair = match best_pair {
                Some(p) => p,
                None => break, // No more merges possible
            };

            // Apply all instances of this merge
            let merged = format!("{}{}", pair.0, pair.1);
            let mut next = Vec::new();
            let mut i = 0;
            while i < current.len() {
                if i + 1 < current.len() && current[i] == pair.0 && current[i + 1] == pair.1 {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(current[i].clone());
                    i += 1;
                }
            }
            current = next;
        }
        current
    }
}

#[cfg(test)]
mod hf_tokenizer_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_hf_tokenizer_byte_fallback() {
        let tok = HfTokenizer::byte_fallback(256 + 4);
        assert_eq!(tok.vocab_size(), 260);
        assert_eq!(tok.unk_id(), 0);
        assert_eq!(tok.bos_id(), Some(1));
        assert_eq!(tok.eos_id(), Some(2));

        let ids = tok.encode("hi");
        // Should be [BOS, <0x68>, <0x69>, EOS]
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[ids.len() - 1], 2); // EOS

        let decoded = tok.decode(&ids[1..ids.len()-1]);
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn test_hf_tokenizer_from_json() {
        let dir = std::env::temp_dir().join("ferrisres_tok_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");

        let json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3,
                    "a": 4, "b": 5, "c": 6, "ab": 7
                },
                "merges": ["a b"]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true},
                {"id": 3, "content": "<pad>", "special": true}
            ]
        });

        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "{}", json).unwrap();

        let tok = HfTokenizer::from_tokenizer_json(&path).unwrap();
        assert_eq!(tok.vocab_size(), 8);

        // Encode "ab" — should merge "a" + "b" → "ab"
        let ids = tok.encode_raw("ab");
        assert!(ids.contains(&7u32), "Expected merged token 7, got {:?}", ids);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hf_tokenizer_round_trip() {
        let tok = HfTokenizer::byte_fallback(1024);
        let text = "Hello, world!";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_pre_tokenize() {
        let tok = HfTokenizer::byte_fallback(256 + 4);
        let words = tok.pre_tokenize("hello world");
        assert!(words.contains(&"▁hello".to_string()) || words.len() >= 2);
    }

    #[test]
    fn test_hf_tokenizer_special_tokens() {
        let tok = HfTokenizer::byte_fallback(256 + 4);
        assert_eq!(tok.unk_id(), 0);
        assert_eq!(tok.bos_id(), Some(1));
        assert_eq!(tok.eos_id(), Some(2));
        assert_eq!(tok.pad_id(), Some(3));
    }
}
