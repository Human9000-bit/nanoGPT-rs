#![forbid(unsafe_code)]

use burn::{
    backend::{self, Autodiff, Wgpu},
    optim::AdamWConfig,
};
use data::{DbPediaDataset, DvachDataset};
use inits::init_train_config;
use log::info;
use tokenizers::Tokenizer;
use train::train;

pub mod args;
pub mod config;
pub mod data;
pub mod inits;
pub mod model;
pub mod train;
pub mod inference;

fn main() -> Result<(), anyhow::Error> {
    // load tokenizer and get vocabulary size
    let vocab = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let vocab_size = vocab.get_vocab_size(false);
    println!("{vocab_size}");

    // backend and autodiff types initialization
    type MyBackend = Wgpu<f32, i32, backend::wgpu::WgslCompiler>;
    type AutoDiff = Autodiff<MyBackend>;

    // device initialization
    let device = backend::wgpu::WgpuDevice::BestAvailable;

    let config = config::parse_config(); //initialize model config from config.toml

    info!("gpt config loaded");

    let artifact_dir = "model/";
    let optimizer = AdamWConfig::new();

    // initialize datasets
    let dataset_train = DvachDataset::train();
    let dataset_test = DbPediaDataset::test();

    // configs initialization
    let gpt_config = inits::init_gpt_config(config["model"].as_table().unwrap(), vocab_size);
    let train_config = init_train_config(config["train"].as_table().unwrap());

    // train the model
    // TODO: add choice between train and inference modes
    train::<AutoDiff, DvachDataset, DbPediaDataset>(
        device,
        dataset_train,
        dataset_test,
        train_config,
        artifact_dir,
        gpt_config,
        optimizer,
    );

    //infer::<MyBackend>(gpt_config, device, Some("primitive".into()));

    Ok(())
}
