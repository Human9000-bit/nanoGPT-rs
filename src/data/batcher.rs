use std::sync::Arc;

use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::Tensor};
use derive_new::new;
use nn::attention::generate_padding_mask;

use super::{TextGenerationItem, Tokenizer};

#[derive(Clone, new)]
pub struct GptBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    max_seq_len: usize,
}

#[derive(Debug, Clone, new)]
pub struct GptBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
pub struct TrainGptBatch<B: Backend> {
    pub tokens_input: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<TextGenerationItem, GptBatch<B>> for GptBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> GptBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text, true))
        }

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
