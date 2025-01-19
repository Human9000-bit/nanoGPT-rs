use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use core::f64;
use nn::{
    attention::generate_autoregressive_mask,
    loss::CrossEntropyLossConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    Embedding, EmbeddingConfig, Linear, LinearConfig,
};

use crate::data::{GptBatch, TrainGptBatch};

#[derive(Config)]
/// Gpt configuration struct
pub struct GptConfig {
    /// Transformer encoder config
    pub transformer: TransformerEncoderConfig,
    /// Vocabulary size from tokenizer
    pub vocab_size: usize,
    /// Padding token from tokenizer
    pub pad_token: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout value
    dropout: f64,
    /// Whether to use bias or not
    bias: bool,
    /// Whether to use quiet softmax instead of the default one or not
    ///
    /// Quiet softmax described in Evan Miller's paper
    /// [Attention is all by one](https://www.evanmiller.org/attention-is-off-by-one.html)
    quiet_softmax: bool,
}

/// The GPT-2 - like model
#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    /// Model transformer
    transformer: TransformerEncoder<B>,
    /// Embedding token encoding
    embd_token: Embedding<B>,
    /// Embedding positional encoding
    embd_pos: Embedding<B>,
    /// Linear layer after transformer pass
    linear: Linear<B>,
    /// Model vocabulary size
    ///
    /// You will probably want to use tokenizer 's vocabulary size for it
    vocab_size: usize,
    /// Index of the padding token
    pad_token: usize,
    /// Max text length model will be able to generate
    max_seq_len: usize,
}

impl GptConfig {
    /// Initialize the actual model from its config
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt<B> {
        // initialize linear layer
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size)
            .with_bias(self.bias)
            .init(device);

        // initialize transformer
        let transformer = self
            .transformer
            .clone()
            .with_dropout(self.dropout)
            .with_quiet_softmax(self.quiet_softmax)
            .init(device);

        // Initialize token embedding...
        let embd_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);

        // ... and positional embedding
        let embd_pos =
            EmbeddingConfig::new(self.max_seq_len, self.transformer.d_model).init(device);

        Gpt {
            transformer,
            embd_token,
            embd_pos,
            linear: output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_len: self.max_seq_len,
        }
    }
}

impl<B: Backend> Gpt<B> {
    #[inline]
    /// Forward pass of the model
    pub fn forward(&self, batch: GptBatch<B>) -> Tensor<B, 2, Float> {
        let [batch_size, seq_len] = batch.tokens.dims();
        let device = &batch.tokens.device();

        // move data to device
        let inputs = batch.tokens.to_device(device);
        let mask = batch.mask_pad.to_device(device);

        let idx_pos: Tensor<B, 2, Int> = Tensor::arange(0..(seq_len as i64), device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);

        // generate positional and token embeddings
        let embd_pos = self.embd_pos.forward(idx_pos);
        let embd_tokens = self.embd_token.forward(inputs);
        let embedding = (embd_pos + embd_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_len, device);

        // the forward pass of the transformer
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask)
                .mask_attn(mask_attn),
        );

        // pass the output through the linear layer
        let output = self.linear.forward(encoded);

        output.reshape([batch_size * seq_len, self.vocab_size])
    }

    #[inline]
    /// Forward pass of the model for classification
    ///
    /// This function also calculates loss, use `.forward()` for inference
    pub fn forward_class(&self, item: TrainGptBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = item.tokens_input.dims();

        // model forward pass
        let batch = GptBatch::new(item.tokens_input, item.mask_pad);
        let outputs = self.forward(batch);
        let targets_flat = item.targets.reshape([batch_size * seq_len]);

        // Calculate cross-entropy loss
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&outputs.device());
        let loss = loss.forward(outputs.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, outputs, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<TrainGptBatch<B>, ClassificationOutput<B>> for Gpt<B> {
    fn step(&self, item: TrainGptBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_class(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainGptBatch<B>, ClassificationOutput<B>> for Gpt<B> {
    fn step(&self, item: TrainGptBatch<B>) -> ClassificationOutput<B> {
        self.forward_class(item)
    }
}
