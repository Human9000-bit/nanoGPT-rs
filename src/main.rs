#![forbid(unsafe_code)]

use burn::{
    backend::{self, Autodiff, Wgpu},
    optim::AdamWConfig,
};
use data::DbPediaDataset;
use inits::init_train_config;
use log::info;
use tokenizers::Tokenizer;
use train::train;

pub mod config;
pub mod data;
pub mod inits;
pub mod model;
pub mod train;

fn main() -> Result<(), anyhow::Error> {
    let vocab = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let vocab_size = vocab.get_vocab_size(false);
    println!("{vocab_size}");

    type MyBackend = Wgpu<f32, i32, backend::wgpu::WgslCompiler>;
    type AutoDiff = Autodiff<MyBackend>;

    let device = backend::wgpu::WgpuDevice::BestAvailable;

    let config = config::parse_config(); //initialize model config from config.toml

    info!("gpt config loaded");

    let artifact_dir = "model/";
    let optimizer = AdamWConfig::new();

    let dataset_train = DbPediaDataset::train();
    let dataset_test = DbPediaDataset::test();

    let gpt_config = inits::init_gpt_config(config["model"].as_table().unwrap(), vocab_size);

    let train_config = init_train_config(config["train"].as_table().unwrap());

    train::<AutoDiff, DbPediaDataset>(
        device,
        dataset_train,
        dataset_test,
        train_config,
        artifact_dir,
        gpt_config,
        optimizer,
    );

    Ok(())
}
