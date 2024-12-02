use rayon::prelude::*;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn start_token(&self) -> usize;
    fn end_token(&self) -> usize;
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
    fn start_token_value(&self) -> String {
        self.decode(&[self.start_token()])
    }
    fn end_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
}

pub struct GptTokenizer {
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
        tokens.get_ids().into_par_iter().map(|t| *t as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.into_par_iter().map(|t| *t as u32).collect::<Vec<u32>>();
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

#[cfg(test)]
mod tests {
    use rand::{Rng, RngCore};

    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = GptTokenizer::default();
        let mut thread_rng = rand::thread_rng();
        let mut sequence = vec![0u8; thread_rng.gen_range(0..10000)];
        thread_rng.fill_bytes(&mut sequence);
        let text: String = sequence.par_iter().map(|i| *i as char).collect();

        let encoded = tokenizer.encode(text.as_str(), false);
        let decoded = tokenizer.decode(encoded.as_slice());

        assert_eq!(text, decoded)
    }
}
