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
