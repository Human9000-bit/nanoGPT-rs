use burn::record::DefaultRecorder;
use burn::{data::dataloader::batcher::Batcher, module::Module};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::path::PathBuf;
use std::sync::Arc;

use burn::prelude::Backend;

use crate::data::Tokenizer;
use crate::{
    data::{GptBatcher, TikTokenizer, TextGenerationItem},
    model::GptConfig,
};

/// Inference the model
///
/// Currently WIP
pub fn infer<B: Backend>(
    model: GptConfig,
    device: B::Device,
    init_prompt: Option<String>,
    artifact_dir: PathBuf,
) {
    let tokenizer = Arc::new(TikTokenizer::default());
    let batcher = GptBatcher::new(tokenizer.clone(), model.max_seq_len);
    let init_prompt = init_prompt.unwrap_or_default();
    let model = model
        .init::<B>(&device)
        .load_file(artifact_dir.join("gpt"), &DefaultRecorder::new(), &device)
        .unwrap_or_else(|_| model.init::<B>(&device));

    let mut context = vec![TextGenerationItem::new(init_prompt)];

    let sequence = model.forward(batcher.batch(context.clone(), &device));

    let text = tokenizer.clone().decode(
        sequence
            .int()
            .to_data()
            .to_vec::<i32>()
            .unwrap()
            .par_iter()
            .map(|i| *i as usize)
            .collect::<Vec<usize>>()
            .as_slice(),
    );

    context.push(TextGenerationItem::new(text.clone()));

    print!("{text}");
}
