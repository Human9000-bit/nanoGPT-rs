use crate::{
    data::{GptBatcher, GptTokenizer, TextGenerationItem, Tokenizer, TrainGptBatch},
    model::GptConfig,
};
use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::{transform::SamplerDataset, Dataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    optim::AdamWConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, CudaMetric, LearningRateMetric,
            LossMetric,
        },
        LearnerBuilder,
    },
};
use log::{info, warn};
use std::{path::PathBuf, sync::Arc};

#[derive(Config)]
pub struct GPTtrainingConfig {
    /// Number of epochs (times to iterate over the whole dataset)
    #[config(default = 1)]
    pub num_epochs: usize,
    /// Real batch size (items per iteration)
    #[config(default = 6)]
    pub batch_size: usize,
    /// Target batch size
    #[config(default = 32)]
    pub target_batch_size: usize,
    /// Number of workers
    #[config(default = 10)]
    pub num_workers: usize,
    /// Random seed
    #[config(default = 792645020)]
    pub seed: u64,
    /// Learning rate
    #[config(deafult = 1.0e-4)]
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    #[config(default = 0.001)]
    pub weight_decay: f64,
    /// Number of warmup steps (linearly increases learning rate from 0 to lr)
    #[config(default = 4000)]
    pub warmup_steps: usize,
}

/// Trains the model
pub fn train<
    B: AutodiffBackend,
    D: Dataset<TextGenerationItem> + 'static,
    DV: Dataset<TextGenerationItem> + 'static,
>(
    device: B::Device,
    dataset_train: D,
    dataset_test: DV,
    config: GPTtrainingConfig,
    artifact_dir: PathBuf,
    gpt_config: GptConfig,
    optimizer: AdamWConfig,
) {
    let artifact_dir = artifact_dir.as_path();
    let tokenizer = Arc::new(GptTokenizer::default());
    let batcher_train = GptBatcher::new(tokenizer.clone(), gpt_config.max_seq_len);
    let batcher_test = GptBatcher::new(tokenizer.clone(), gpt_config.max_seq_len);

    B::seed(config.seed);

    // Load model weights
    let model = gpt_config.init::<B>(&device);
    let model = model
        .load_file("model/gpt", &DefaultRecorder::new(), &device)
        .unwrap_or_else(|_| gpt_config.init::<B>(&device));

    // Save model config
    let save_res = config.save(artifact_dir.join("config.json"));
    match save_res {
        Ok(()) => info!("model config saved"),
        Err(e) => warn!("failed to save model config; {e}"),
    }

    // Initialize dataloaders
    // train
    let dataloader_train: Arc<dyn DataLoader<TrainGptBatch<B>>> =
        DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.seed)
            .build(SamplerDataset::new(dataset_train, 10_000));

    // test
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(SamplerDataset::new(dataset_test, 1000));

    // gradient accumulation
    let accum = config.target_batch_size / config.batch_size;

    // optimizer initialization
    let optim = optimizer
        .with_weight_decay(config.weight_decay as f32)
        .init();

    // learning rate scheduler initialization
    // noam lr scheduler increases learning rate linearly from 0 to lr
    // and then decreases it exponentially
    let lr_sched = NoamLrSchedulerConfig::new(config.learning_rate / accum as f64)
        .with_warmup_steps(config.warmup_steps)
        .with_model_size(gpt_config.transformer.d_model)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train(CpuUse::new())
        .metric_train(CpuMemory::new())
        .metric_train(CpuTemperature::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optim, lr_sched);

    //thread::sleep(Duration::from_secs(5));

    // train the model
    let model = learner.fit(dataloader_train, dataloader_test);

    // save the model
    let _ = model
        .save_file(artifact_dir.join("gpt"), &DefaultRecorder::new())
        .inspect_err(|e| log::error!("failed to save the model: {e}"));
}
