use std::sync::Arc;

use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::Tensor};
use derive_new::new;
use nn::attention::generate_padding_mask;
use rayon::prelude::*;

use super::{TextGenerationItem, Tokenizer};

#[derive(Clone, new)]
/// Batcher for the GPT model
pub struct GptBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    max_seq_len: usize,
}

#[derive(Debug, Clone, new)]
/// Evaluation batch for the GPT model
pub struct GptBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
/// Training batch for the GPT model with targets
pub struct TrainGptBatch<B: Backend> {
    pub tokens_input: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<TextGenerationItem, GptBatch<B>> for GptBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> GptBatch<B> {
        // encode the items in parallel
        let tokens_list = items
            .into_par_iter()
            .map(|item| self.tokenizer.encode(&item.text, true))
            .collect();
        // generate the padding mask
        let mask: nn::attention::GeneratePaddingMask<B> = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_len),
            &B::Device::default(),
        );

        GptBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<TextGenerationItem, TrainGptBatch<B>> for GptBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TrainGptBatch<B> {
        let item: GptBatch<B> = self.batch(items);
        let [batch_size, seq_length] = item.tokens.dims();

        let inputs = item
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TrainGptBatch::new(inputs, targets, mask_pad)
    }
}
