
use burn::nn::transformer::TransformerEncoderConfig;

use crate::{
    data::{GptTokenizer, Tokenizer},
    model::GptConfig,
    train::GPTtrainingConfig,
};

pub fn init_gpt_config(
    config: &toml::map::Map<String, toml::Value>,
    vocab_size: usize,
) -> GptConfig {
    let d_model = config["n_embd"].as_integer().unwrap() as usize;
    let n_heads = config["n_head"].as_integer().unwrap() as usize;
    let n_layers = config["n_layer"].as_integer().unwrap() as usize;
    let d_ff_k = config["d_ff_k"].as_integer().unwrap() as usize;
    let dropout = config["dropout"].as_float().unwrap();
    let bias = config["bias"].as_bool().unwrap();
    let max_seq_len = config["max_seq_len"].as_integer().unwrap() as usize;

    let pad_token = GptTokenizer::default().pad_token();

    let transf = TransformerEncoderConfig::new(d_model, d_model * d_ff_k, n_heads, n_layers);
    GptConfig::new(transf, vocab_size, pad_token, max_seq_len, dropout, bias)
}

pub fn init_train_config(config: &toml::map::Map<String, toml::Value>) -> GPTtrainingConfig {
    let num_epochs = config["num_epochs"].as_integer().unwrap() as usize;
    let batch_size = config["batch_size"].as_integer().unwrap() as usize;
    let target_batch_size = config["target_batch_size"].as_integer().unwrap() as usize;
    let num_workers = config["num_workers"].as_integer().unwrap() as usize;
    let seed = config["seed"].as_integer().unwrap() as u64;
    let learning_rate = config["learning_rate"].as_float().unwrap();
    let weight_decay = config["weight_decay"].as_float().unwrap();
    let warmup_steps = config["warmup_steps"].as_integer().unwrap() as usize;
    assert!(target_batch_size % batch_size == 0 && target_batch_size > batch_size);
    
    GPTtrainingConfig {
        num_epochs,
        batch_size,
        target_batch_size,
        num_workers,
        seed,
        learning_rate,
        weight_decay,
        warmup_steps
    }
}
