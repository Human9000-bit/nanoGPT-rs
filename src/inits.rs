use crate::model;

pub fn init_model_config(config: &toml::map::Map<String, toml::Value>, vocab_size: usize) -> model::ModelConfig {
    model::ModelConfig::new(
        config["n_head"].as_integer().unwrap() as usize,
        config["n_embd"].as_integer().unwrap() as usize,
        config["bias"].as_bool().unwrap(),
        config["dropout"].as_float().unwrap(),
        config["n_layer"].as_integer().unwrap() as usize,
        config["block_size"].as_integer().unwrap() as usize,
        vocab_size)
}

pub fn init_train_config() {}