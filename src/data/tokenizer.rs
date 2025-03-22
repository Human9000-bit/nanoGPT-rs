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

/// [crate::model::Gpt]'s tokenizer
pub struct GptTokenizer {
    /// Tokenizer for the model
    tokenizer: tokenizers::Tokenizer,
}

impl Default for GptTokenizer {
    fn default() -> Self {
        let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None).unwrap();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("[START]", true),
            tokenizers::AddedToken::from("[END]", true),
            tokenizers::AddedToken::from("[PAD]", true),
        ]);
        Self { tokenizer }
    }
}

impl Tokenizer for GptTokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        let text = match special_tokens {
            true => format!("[START]{}[END]", value),
            false => value.to_string(),
        };

        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens
            .get_ids()
            .into_par_iter()
            .map(|t| *t as usize)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens
            .into_par_iter()
            .map(|t| *t as u32)
            .collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }

    fn start_token(&self) -> usize {
        self.tokenizer.token_to_id("[START]").unwrap() as usize
    }

    fn end_token(&self) -> usize {
        self.tokenizer.token_to_id("[END]").unwrap() as usize
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
        Self::new(tiktoken_rs::r50k_base().unwrap())
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
    
    fn start_token_value(&self) -> String {
        todo!()
    }
    
    fn start_token(&self) -> usize {
        todo!()
    }
    
    fn end_token(&self) -> usize {
        todo!()
    }
    
    fn end_token_value(&self) -> String {
        todo!()
    }
    
    fn pad_token_value(&self) -> String {
        todo!()
    }
    
    fn pad_token(&self) -> usize {
        todo!()
    }
    
    fn vocab_size(&self) -> usize {
        // i found it in tiktoken source code
        50256
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    quickcheck! {
        /// Test that random string is the same as encoded and decoded back
        fn test_encode_decode(text: String) -> bool {
            let tokenizer = GptTokenizer::default();

            let encoded = tokenizer.encode(text.as_str(), false);
            let decoded = tokenizer.decode(encoded.as_slice());

            text == decoded
        }
    }
}
