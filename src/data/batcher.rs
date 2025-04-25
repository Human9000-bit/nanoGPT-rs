use std::sync::Arc;

use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::Tensor};
use derive_new::new;
use nn::attention::generate_padding_mask;
use rayon::prelude::*;

use super::{TextGenerationItem, Tokenizer};

#[derive(Clone, new)]
/// Batcher for the GPT model
pub struct GptBatcher<T: Tokenizer> {
    /// Batcher 's pointer to tokenizer
    tokenizer: Arc<T>,
    /// Maximal sequence length of batch text
    max_seq_len: usize,
}

#[derive(Debug, Clone, new)]
/// Evaluation batch for the GPT model
pub struct GptBatch<B: Backend> {
    /// Tokenized text of the batch
    pub tokens: Tensor<B, 2, Int>,
    /// Padding mask of that text
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
/// Training batch for the GPT model with targets
pub struct TrainGptBatch<B: Backend> {
    /// Input tokens passed to model
    pub tokens_input: Tensor<B, 2, Int>,
    /// Padding mask for input tokens
    pub mask_pad: Tensor<B, 2, Bool>,
    /// Targets aka labels
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend, T: Tokenizer> Batcher<B, TextGenerationItem, GptBatch<B>> for GptBatcher<T> {
    fn batch(&self, items: Vec<TextGenerationItem>, device: &B::Device) -> GptBatch<B> {
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
            device,
        );

        GptBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend, T: Tokenizer> Batcher<B, TextGenerationItem, TrainGptBatch<B>> for GptBatcher<T> {
    fn batch(&self, items: Vec<TextGenerationItem>, device: &B::Device) -> TrainGptBatch<B> {
        let item: GptBatch<B> = self.batch(items, device);
        let [batch_size, seq_length] = item.tokens.dims();

        let inputs = item
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TrainGptBatch::new(inputs, mask_pad, targets)
    }
}
