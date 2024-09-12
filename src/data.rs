use burn::{data::dataloader::batcher::Batcher, prelude::*};
use tokenizers::Tokenizer;
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

pub struct GptBatch<B: Backend> {
    pub text: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<String, GptBatch<B>> for GptBatcher<B>  {
    fn batch(&self, items: Vec<String>) -> GptBatch<B> {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let batches: Vec<Tensor<B, 1, Int>> = items.par_iter()
            .map(|string| tokenizer.encode(string.to_string(), false).unwrap())
            .map(|idx| idx.get_ids().to_owned().par_iter().map(|&f| f as i32).collect())
            .map(|vec: Vec<i32>| Tensor::from_ints(vec.as_slice(), &self.device)).collect();
        let tensor =  Tensor::stack(batches, 0);
        GptBatch { text: tensor }
    }
}