use std::{fs::File, io::Read, sync::Arc, thread, time::Duration};

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
    record::{
        CompactRecorder, DefaultRecorder, HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder,
    },
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, CudaMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use log::{info, warn};

#[derive(Config)]
pub struct GPTtrainingConfig {
    #[config(default = 1)]
    pub num_epochs: usize,

    #[config(default = 6)]
    pub batch_size: usize,

    #[config(default = 32)]
    pub target_batch_size: usize,

    #[config(default = 10)]
    pub num_workers: usize,

    #[config(default = 792645020)]
    pub seed: u64,

    #[config(deafult = 1.0e-4)]
    pub learning_rate: f64,

    #[config(default = 0.001)]
    pub weight_decay: f64,

    #[config(default = 4000)]
    pub warmup_steps: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: GPTtrainingConfig,
    artifact_dir: &str,
    gpt_config: GptConfig,
    optimizer: AdamWConfig,
) {
    let tokenizer = Arc::new(GptTokenizer::default());
    let batcher_train = GptBatcher::new(tokenizer.clone(), gpt_config.max_seq_len);
    let batcher_test = GptBatcher::new(tokenizer.clone(), gpt_config.max_seq_len);

    B::seed(config.seed);
    
    let mut buf = Vec::new();
    let _ = File::open("model/gpt.mpk").unwrap().read_to_end(&mut buf); 

    let record = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new().load(buf, &device);

    let model = match record {
        Ok(record) => {
            info!("model weights loaded");
            gpt_config.init::<B>(&device).load_record(record)
        }
        Err(_) => gpt_config.init::<B>(&device),
    };

    let save_res = config.save(format!("{artifact_dir}/config.json"));
    match save_res {
        Ok(()) => info!("model config saved"),
        Err(e) => warn!("failed to save model config; {e}"),
    }

    let dataloader_train: Arc<dyn DataLoader<TrainGptBatch<B>>> =
        DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.seed)
            .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(SamplerDataset::new(dataset_test, 1000));

    let accum = config.target_batch_size / config.batch_size;
    let optim = optimizer
        .with_weight_decay(config.weight_decay as f32)
        .init();
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

    thread::sleep(Duration::from_secs(5));

    let model = learner.fit(dataloader_train, dataloader_test);

    let _ = model
        .save_file("model/gpt", &DefaultRecorder::new())
        .inspect_err(|e| log::error!("failed to save the model: {e}"));
}
