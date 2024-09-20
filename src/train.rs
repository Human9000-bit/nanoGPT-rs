use std::rc::Rc;
pub(crate) use std::fs;

use burn::
    {data::{dataloader::DataLoaderBuilder,dataset::HuggingfaceDatasetLoader},
    module::Module,nn::loss::CrossEntropyLossConfig, optim::AdamConfig, prelude::*,
    record::CompactRecorder, tensor::backend::AutodiffBackend, 
    train::{metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep}
};

use crate::{data::{GptBatch, GptBatcher}, model::{GPTConfig, GPT}};

impl<B: Backend> GPT<B> {
    pub fn forward_regression(
        &self,
        batch: GptBatch<B>) 
    -> RegressionOutput<B> {
        let batch_targets_dims = batch.targets.dims();
        let batch_device = &batch.text.device();
        let output = self.forward(batch.text);
        assert_eq!(output.dims()[0], batch_targets_dims[0]);
        let cel = CrossEntropyLossConfig::new()
            .init(batch_device);
        let loss = cel.forward(output.clone().flatten(1, 2), batch.targets.clone());
        
        RegressionOutput::new(loss, output.flatten(0, 1), batch.targets.float().unsqueeze())
    }
}

impl<B: AutodiffBackend> TrainStep<GptBatch<B>, RegressionOutput<B>> for GPT<B> {
    fn step(&self, item: GptBatch<B>) -> burn::train::TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(item);
        
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GptBatch<B>, RegressionOutput<B>> for GPT<B> {
    fn step(&self, item: GptBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item)
    }
}

#[derive(Config)]
pub struct GPTtrainingConfig {
    pub model: GPTConfig,
    pub optimizer: AdamConfig,
    
    #[config(default = 10)]
    pub num_epochs: usize,
    
    #[config(default = 64)]
    pub batch_size: usize,
    
    #[config(dfault = 4)]
    pub num_workers: usize,
    
    #[config(default = 792645020)]
    pub seed: u64,
    
    #[config(deafult = 1.0e-4)]
    pub learning_rate: f64
}

pub fn create_artifact_dir(dir: &str) {
    fs::remove_dir_all(dir).ok();
    fs::create_dir_all(dir).ok();
}

pub fn train<B: AutodiffBackend>(artifacts_dir: &str, config: GPTtrainingConfig, device: B::Device) {
    let device = Rc::new(device);
    create_artifact_dir(artifacts_dir);
    config
        .save(format!("{artifacts_dir}/config.json"))
        .expect("failed to create artifact dir");
    
    B::seed(config.seed);
    
    let loader = HuggingfaceDatasetLoader::new("fka/awesome-chatgpt-prompts");
    
    let train_batcher = GptBatcher::<B>::new(Rc::unwrap_or_clone(device.clone()));
    let train_valid = GptBatcher::<B::InnerBackend>::new(Rc::unwrap_or_clone(device.clone()));
    
    let train_dataloader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(loader.dataset("train").unwrap());
    
    let loader = HuggingfaceDatasetLoader::new("fka/awesome-chatgpt-prompts");
    
    println!("train datasel loaded");
    
    let test_dataloader = DataLoaderBuilder::new(train_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(loader.dataset("train").unwrap());
    
    println!("test dataset loaded");
    
    let device = Rc::unwrap_or_clone(device);
    
    let learner = LearnerBuilder::new(artifacts_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(config.model.init::<B>(&device), config.optimizer.init(), config.learning_rate);
    
    let trained_model = learner.fit(train_dataloader, test_dataloader);
    
    trained_model
        .save_file(format!("{artifacts_dir}/model"), &CompactRecorder::new())
        .expect("failed to save the model");
}