use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    next_id: u32,
    // special_tokens: HashMap<String, u32>,
}

impl Tokenizer {
    // word-level tokenization
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_id: 0,
        };

        // add the most common special tokens first
        tokenizer.add_token("<PAD>"); // id=0
        tokenizer.add_token("<UNK>"); // id=1
        tokenizer.add_token("<BOS>"); // id=2
        tokenizer.add_token("<EOS>"); // id=3

        tokenizer
    }

    fn add_token(&mut self, token: &str) -> u32 {
        let id = self.next_id;
        self.vocab.insert(token.to_string(), id);
        self.reverse_vocab.insert(id, token.to_string());
        self.next_id += 1;
        id
    }

    pub fn build_vocab(&mut self, texts: &[&str]) {
        for text in texts {
            for word in text.split_whitespace() {
                let clean_word = word.to_lowercase();
                if !self.vocab.contains_key(&clean_word) {
                    self.add_token(&clean_word);
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let clean_word = word.to_lowercase();

            if let Some(&token_id) = self.vocab.get(&clean_word) {
                tokens.push(token_id);  // found in vocabulary
            } else {
                // unknown words witth the UNK token
                tokens.push(self.vocab["<UNK>"]);
            }
        }
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut words = Vec::new();

        for &token_id in tokens {
            if let Some(word) = self.reverse_vocab.get(&token_id) {
                words.push(word.clone());
            } else {
                words.push("<UNK>".to_string());  // missing token IDs
            }
        }

        words.join(" ")
    }
}
