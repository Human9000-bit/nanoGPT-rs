use tokenizers::Tokenizer;

pub mod model;
pub mod ops;
pub mod data;
pub mod config;

fn main() {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None);
    
}