#![forbid(unsafe_code)]
#![recursion_limit = "256"]
#![macro_use]
extern crate log;
extern crate pretty_env_logger;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::error::Error;

use burn::{
    backend::{self, Autodiff, Cuda, Wgpu},
    optim::AdamWConfig,
};
use clap::{Parser, Subcommand};
use data::DbPediaDataset;
use inits::init_train_config;
use log::info;
use train::train;

use crate::data::{TikTokenizer, Tokenizer};

pub mod config;
pub mod data;
pub mod inference;
pub mod inits;
pub mod model;
pub mod train;

#[derive(Parser)]
#[clap(name = "nanogpt-rs")]
#[clap(about = "A NanoGPT implementation in Rust using Burn")]
struct Cli {
    /// Backend to use for computation
    #[clap(short, long, default_value = "wgpu")]
    backend: Backend,
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Backend {
    /// Use WGPU backend (default, cross-platform)
    Wgpu,
    /// Use CUDA backend (NVIDIA GPUs only)
    Cuda,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model
    Train,
    /// Run inference with the model
    Infer {
        /// Optional prompt for inference
        #[clap(short, long, default_value = "primitive")]
        prompt: String,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    let cli = Cli::parse();
    // load tokenizer and get vocabulary size
    let vocab = TikTokenizer::default();
    let vocab_size = vocab.vocab_size();
    println!("{vocab_size}");

    let config = config::parse_config(); // initialize model config from config.toml
    info!("gpt config loaded");

    let artifact_dir = std::path::PathBuf::from("model/");
    let gpt_config = inits::init_gpt_config(config["model"].as_table().unwrap(), vocab_size);

    // backend and device initialization based on CLI argument
    match cli.backend {
        Backend::Cuda => run_with_cuda_backend(cli.command, gpt_config, artifact_dir, config),
        Backend::Wgpu => run_with_wgpu_backend(cli.command, gpt_config, artifact_dir, config),
    }

    Ok(())
}

fn run_with_cuda_backend(
    command: Commands,
    gpt_config: crate::model::GptConfig,
    artifact_dir: std::path::PathBuf,
    config: toml::Table,
) {
    type MyBackend = Cuda;
    let device = backend::cuda::CudaDevice::new(0);

    match command {
        Commands::Train => {
            let optimizer = AdamWConfig::new();

            // initialize datasets
            let dataset_train = DbPediaDataset::train();
            let dataset_test = DbPediaDataset::test();

            // configs initialization
            let train_config = init_train_config(config["train"].as_table().unwrap());

            // train the model
            type AutoDiff = Autodiff<MyBackend>;
            train::<AutoDiff, DbPediaDataset, DbPediaDataset>(
                device,
                dataset_train,
                dataset_test,
                train_config,
                artifact_dir,
                gpt_config,
                optimizer,
            );
        }
        Commands::Infer { prompt } => {
            inference::infer::<MyBackend>(gpt_config, device, Some(prompt), artifact_dir);
        }
    }
}

fn run_with_wgpu_backend(
    command: Commands,
    gpt_config: crate::model::GptConfig,
    artifact_dir: std::path::PathBuf,
    config: toml::Table,
) {
    type MyBackend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    match command {
        Commands::Train => {
            let optimizer = AdamWConfig::new();

            // initialize datasets
            let dataset_train = DbPediaDataset::train();
            let dataset_test = DbPediaDataset::test();

            // configs initialization
            let train_config = init_train_config(config["train"].as_table().unwrap());

            // train the model
            type AutoDiff = Autodiff<MyBackend>;
            train::<AutoDiff, DbPediaDataset, DbPediaDataset>(
                device,
                dataset_train,
                dataset_test,
                train_config,
                artifact_dir,
                gpt_config,
                optimizer,
            );
        }
        Commands::Infer { prompt } => {
            inference::infer::<MyBackend>(gpt_config, device, Some(prompt), artifact_dir);
        }
    }
}
