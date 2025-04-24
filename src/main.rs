#![forbid(unsafe_code)]

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::path::Path;

use burn::{
    backend::{self, Autodiff, Cuda},
    optim::AdamWConfig,
};
use data::DbPediaDataset;
use inits::init_train_config;
use log::info;
use tokenizers::Tokenizer;
use train::train;

pub mod config;
pub mod data;
pub mod inference;
pub mod inits;
pub mod model;
pub mod train;

fn main() -> Result<(), anyhow::Error> {
    let command = std::env::args().nth(1).unwrap();
    // load tokenizer and get vocabulary size
    let vocab = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let vocab_size = vocab.get_vocab_size(false);
    println!("{vocab_size}");

    // backend and autodiff types initialization
    type MyBackend = Cuda;

    // device initialization
    let device = backend::cuda::CudaDevice::new(0);

    let config = config::parse_config(); //initialize model config from config.toml

    info!("gpt config loaded");

    let artifact_dir = Path::new("model/");
    let gpt_config = inits::init_gpt_config(config["model"].as_table().unwrap(), vocab_size);

    match command.as_str() {
        "train" => {
            let optimizer = AdamWConfig::new();

            // initialize datasets
            let dataset_train = DbPediaDataset::train();
            let dataset_test = DbPediaDataset::test();

            // configs initialization
            let train_config = init_train_config(config["train"].as_table().unwrap());

            // train the model
            // TODO: add choice between train and inference modes
            type AutoDiff = Autodiff<MyBackend>;
            train::<AutoDiff, DbPediaDataset, DbPediaDataset>(
                device,
                dataset_train,
                dataset_test,
                train_config,
                artifact_dir.to_path_buf(),
                gpt_config,
                optimizer,
            );
        }
        &_ => inference::infer::<MyBackend>(
            gpt_config,
            device,
            Some("primitive".into()),
            artifact_dir.to_path_buf(),
        ),
    }

    Ok(())
}
