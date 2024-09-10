use std::error::Error;

use burn::{backend::{self, Autodiff, Wgpu}, tensor::{Int, Tensor}};
use model::{BlockConfig, MlpConfig, SelfAttentionConfig};
use tokenizers::Tokenizer;
use rayon::prelude::*;

pub mod model;
pub mod ops;
pub mod data;
pub mod config;

fn main() -> Result<(), Box<dyn Error>> {
    let vocab = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let vocab_size = vocab.get_vocab_size(false);
    
    type MyBackend = Wgpu<backend::wgpu::AutoGraphicsApi, f32, i32>;
    type AutoDiff = Autodiff<MyBackend>;
    
    let device = backend::wgpu::WgpuDevice::BestAvailable;
    
    let config = config::parse_config();
    let modelconfig_toml = config["model"].as_table().unwrap();
    let modelconfig = model::ModelConfig::new(
        modelconfig_toml["n_head"].as_integer().unwrap() as usize,
        modelconfig_toml["n_embd"].as_integer().unwrap() as usize,
        modelconfig_toml["bias"].as_bool().unwrap(),
        modelconfig_toml["dropout"].as_float().unwrap(),
        modelconfig_toml["n_layer"].as_integer().unwrap() as usize,
        modelconfig_toml["block_size"].as_integer().unwrap() as usize,
        vocab_size
    );
    
    let block_config = BlockConfig::new(
                SelfAttentionConfig::new(modelconfig.clone()),
                MlpConfig::new(modelconfig.clone().n_embd, modelconfig.clone().dropout, modelconfig.clone().bias), modelconfig.clone());
    
    let gpt: model::GPT<MyBackend> = model::GPTConfig::new(
        model::TranformerConfig::new(
            modelconfig.clone(), 
            block_config.clone()), block_config, modelconfig).init(&device);
    
    let x = vocab.encode("helo", true).unwrap();
    let x: Vec<i32> = x.get_ids().to_owned().par_iter().map(|x| x.to_owned() as i32).collect();
    let x = Tensor::<MyBackend, 1, Int>::from_ints(x.as_slice(), &device);
    let x = gpt.generate(x.unsqueeze(), 4, Some(1.0), None);
    println!("{:?}", x);
    
    Ok(())
}