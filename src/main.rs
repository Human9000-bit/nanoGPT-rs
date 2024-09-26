use std::error::Error;

use burn::{backend::{self, Autodiff, Candle}, optim::AdamConfig, tensor::f16};
use log::info;
use model::{BlockConfig, MlpConfig, SelfAttentionConfig};
use tokenizers::Tokenizer;
use inits::init_model_config;

pub mod model;
pub mod ops;
pub mod data;
pub mod config;
pub mod train;
pub mod inits;

fn main() -> Result<(), Box<dyn Error>> {
    let vocab = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let vocab_size = vocab.get_vocab_size(false);
    
    type MyBackend = Candle<f16, u32>;
    type AutoDiff = Autodiff<MyBackend>;
    
    let device = backend::candle::CandleDevice::Cpu;
    
    let config = config::parse_config(); //initialize model config from config.toml
    let model_config = init_model_config(config["model"].as_table().unwrap(), vocab_size);
    
    let block_config = BlockConfig::new(
                SelfAttentionConfig::new(model_config.clone()),
                MlpConfig::new(
                    model_config.n_embd, 
                    model_config.dropout, 
                    model_config.bias), 
                model_config.clone()
    );
    
    let gpt = model::GPTConfig::new(
        model::TranformerConfig::new(
            model_config.clone(), 
            block_config.clone()), block_config, model_config.clone());
    info!("gpt config loaded");
    
    let artifact_dir = "model/";
    let optim = AdamConfig::new();
    let config = train::GPTtrainingConfig::new(gpt, optim);
    crate::train::train::<AutoDiff>(artifact_dir, config, device);
    
    Ok(())
}