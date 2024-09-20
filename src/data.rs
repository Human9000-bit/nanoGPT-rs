use burn::{data::dataloader::batcher::Batcher, prelude::*};
use tokenizers::{PaddingDirection, Tokenizer};
use rayon::prelude::*;

#[derive(Clone)]
pub struct GptBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> GptBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct GptBatch<B: Backend> {
    pub text: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 1, Int>
}

///batch Vec of Strings into 2 dim tensor of encoded sequence
impl<B: Backend> Batcher<String, GptBatch<B>> for GptBatcher<B>  {
    fn batch(&self, items: Vec<String>) -> GptBatch<B> {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let batches: Vec<Tensor<B, 1, Int>> = items.par_iter()
            .map(|string| tokenizer.encode(string.to_string(), false).unwrap())
            .map(|idx| {let mut idx = idx; idx.pad(12, 1, 2, "", PaddingDirection::Right); idx})
            .map(|idx_pad| idx_pad.get_ids().to_owned().par_iter().map(|&f| f as i32).collect())
            .map(|vec: Vec<i32>| Tensor::from_ints(vec.as_slice(), &self.device)).collect();
        let tensor = Tensor::stack(batches, 0);
        let targets = tensor.clone().slice([0..tensor.dims()[0], tensor.dims()[1]/2..tensor.dims()[1]])
            .flatten(0, 1);
        let logits = tensor.clone().slice([0..tensor.dims()[0], 0..tensor.dims()[1]/2]);
        GptBatch { text: logits, targets }
    }
}