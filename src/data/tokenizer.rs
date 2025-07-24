use rayon::prelude::*;
use tiktoken_rs::CoreBPE;

/// General trait for tokenizer
pub trait Tokenizer: Send + Sync {
    /// Encode a [&str] into Vec of indices
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;

    /// Decode a Vec of indices into [String]
    fn decode(&self, tokens: &[usize]) -> String;

    /// Get a vocabulary size of tokens
    fn vocab_size(&self) -> usize;

    /// Get the index of PAD token
    fn pad_token(&self) -> usize;

    /// Get the index of START token
    fn start_token(&self) -> usize;

    /// Get the index of END token
    fn end_token(&self) -> usize;

    /// Decode the value of the PAD token
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }

    /// Decode the value of the START token
    fn start_token_value(&self) -> String {
        self.decode(&[self.start_token()])
    }

    /// Decode the value of the END token
    fn end_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
}

pub struct TikTokenizer {
    tokenizer: CoreBPE,
}

impl TikTokenizer {
    pub fn new(tokenizer: CoreBPE) -> Self {
        Self { tokenizer }
    }
}

impl Default for TikTokenizer {
    fn default() -> Self {
        Self::new(tiktoken_rs::cl100k_base().unwrap())
    }
}

impl Tokenizer for TikTokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        match special_tokens {
            true => self
                .tokenizer
                .encode_with_special_tokens(value)
                .into_par_iter()
                .map(|int| int as usize)
                .collect(),
            false => self
                .tokenizer
                .encode_ordinary(value)
                .into_par_iter()
                .map(|int| int as usize)
                .collect(),
        }
    }
    
    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.into_par_iter().map(|int| *int as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(tokens).unwrap()
    }
    
    fn start_token(&self) -> usize {
        todo!()
    }
    
    fn start_token_value(&self) -> String {
        todo!()
    }
    
    fn end_token(&self) -> usize {
        100257
    }
    
    fn end_token_value(&self) -> String {
        "<|endoftext|>".to_string()
    }
    
    fn pad_token(&self) -> usize {
        100256
    }
    
    fn pad_token_value(&self) -> String {
        "<|pad|>".to_string()
    }
    
    fn vocab_size(&self) -> usize {
        // i found it in tiktoken source code
        100277
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    quickcheck! {
        /// Test that random string is the same as encoded and decoded back
        fn test_encode_decode_gpt_tokenizer(text: String) -> bool {
            let tokenizer = TikTokenizer::default();

            let encoded = tokenizer.encode(text.as_str(), false);
            let decoded = tokenizer.decode(encoded.as_slice());

            text == decoded
        }
        
        fn test_encode_decode_tiktoken(text: String) -> bool {
            let tokenizer = TikTokenizer::default();

            let encoded = tokenizer.encode(text.as_str(), false);
            let decoded = tokenizer.decode(encoded.as_slice());

            text == decoded
        }
    }
}
