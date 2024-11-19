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
    let config: GPTtrainingConfig = toml::from_str(config.to_string().as_str()).unwrap();
    assert!(
        config.target_batch_size % config.batch_size == 0
            && config.target_batch_size > config.batch_size
    );
    config
}
