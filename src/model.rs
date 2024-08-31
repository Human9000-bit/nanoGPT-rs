use core::f64;

use burn::{nn::attention, prelude::*, tensor::activation::softmax};
use nn::{LayerNormConfig, LinearConfig};

use crate::ops;

#[derive(Config, Debug)]
pub struct ModelConfig {
    n_head: usize,
    n_embd: usize,
    bias: bool,
    dropout: f64,
    vocab_size: usize,
    n_layer: usize,
    block_size: usize,
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    c_attn: nn::Linear<B>,
    c_proj: nn::Linear<B>,
    resid_dropout: nn::Dropout,
    attn: attention::MultiHeadAttention<B>,
    n_embd: usize,
    n_head: usize,
}

#[derive(Config, Debug)]
pub struct SelfAttentionConfig {
    config: ModelConfig,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.config.n_embd % self.config.n_head == 0);
        SelfAttention {
            c_attn: nn::LinearConfig::new(self.config.n_embd, self.config.n_embd * 3)
                .with_bias(self.config.bias)
                .init(device),
            c_proj: nn::LinearConfig::new(self.config.n_embd, self.config.n_embd)
                .with_bias(self.config.bias)
                .init(device),
            resid_dropout: nn::DropoutConfig::new(self.config.dropout).init(),
            attn: attention::MultiHeadAttentionConfig::new(self.config.n_embd, self.config.n_head)
                .with_dropout(self.config.dropout)
                .init(device),
            n_embd: self.config.n_embd,
            n_head: self.config.n_head,
        }
    }
}

impl<B: Backend> SelfAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.dims();
        let (b, t, c) = (shape[0], shape[1], shape[2]);

        let temp = self.c_attn.forward(x).chunk(self.n_embd, 2);
        let (q, k, v) = (temp[0].clone(), temp[1].clone(), temp[2].clone());
        drop(temp);

        let k = k.reshape([b, t, c / self.n_head]).swap_dims(1, 2);
        let q = q.reshape([b, t, c / self.n_head]).swap_dims(1, 2);
        let v = v.reshape([b, t, c / self.n_head]).swap_dims(1, 2);

        let mhainput = attention::MhaInput::new(q, k, v);

        let y = self.attn.forward(mhainput);
        let y = y.context;
        self.resid_dropout.forward(self.c_proj.forward(y))
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: nn::Linear<B>,
    gelu: nn::Gelu,
    c_proj: nn::Linear<B>,
    dropout: nn::Dropout,
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    n_embd: usize,
    dropout: f64,
    bias: bool,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            c_fc: nn::LinearConfig::new(self.n_embd, self.n_embd * 4)
                .with_bias(self.bias)
                .init(device),
            gelu: nn::Gelu::new(),
            c_proj: nn::LinearConfig::new(self.n_embd * 4, self.n_embd)
                .with_bias(self.bias)
                .init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.gelu.forward(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct Block<B: Backend> {
    ln_1: nn::LayerNorm<B>,
    attn: SelfAttention<B>,
    ln_2: nn::LayerNorm<B>,
    mlp: Mlp<B>,
}

#[derive(Debug, Config)]
pub struct BlockConfig {
    attn_config: SelfAttentionConfig,
    mlp_config: MlpConfig,
    config: ModelConfig,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            ln_1: nn::LayerNormConfig::new(self.config.n_embd).init(device),
            attn: self.attn_config.init(device),
            ln_2: LayerNormConfig::new(self.config.n_embd).init(device),
            mlp: self.mlp_config.init(device),
        }
    }
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x));
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        x
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    pub wte: nn::Embedding<B>,
    pub wpe: nn::Embedding<B>,
    pub drop: nn::Dropout,
    pub h: Vec<Block<B>>,
    pub ln_f: nn::LayerNorm<B>,
}

#[derive(Debug, Config)]
pub struct TranformerConfig {
    config: ModelConfig,
    block_config: BlockConfig,
}

impl TranformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let mut blocks = Vec::new();
        let (attn_config, mlp_config, _model_config) = (
            self.block_config.attn_config.clone(),
            self.block_config.mlp_config.clone(),
            self.config.clone(),
        );
        for _ in [0..self.config.n_layer] {
            blocks.push(
                BlockConfig::new(attn_config.clone(), mlp_config.clone(), self.config.clone())
                    .init(device),
            )
        }
        Transformer {
            wte: nn::EmbeddingConfig::new(self.config.vocab_size, self.config.n_embd).init(device),
            wpe: nn::EmbeddingConfig::new(self.config.block_size, self.config.n_embd).init(device),
            drop: nn::DropoutConfig::new(self.config.dropout).init(),
            h: blocks,
            ln_f: LayerNormConfig::new(self.config.n_embd).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    transf: Transformer<B>,
    lm_head: nn::Linear<B>,
    block_size: usize,
}

#[derive(Config, Debug)]
pub struct GPTConfig {
    transf: TranformerConfig,
    block_conf: BlockConfig,
    config: ModelConfig,
}

impl GPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT<B> {
        let transf = self.transf.clone().init(device);

        println!("Number of parameters: {}", &transf.num_params());
        GPT {
            transf,
            lm_head: LinearConfig::new(self.config.n_embd, self.config.vocab_size)
                .with_bias(self.config.bias)
                .init(device),
            block_size: self.config.block_size,
        }
    }
}

impl<B: Backend> GPT<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = idx.device();
        let t = idx.dims()[1];
        assert!(t <= self.block_size);
        let tf = t as i64;
        let pos = Tensor::arange(0..tf, &device);

        let tok_emb = self.transf.wte.forward(idx);
        let pos_emb = self.transf.wpe.forward(pos.unsqueeze());
        let x = self.transf.drop.forward(tok_emb + pos_emb);

        for block in &self.transf.h {
            let x = block.forward(x.clone());
        }

        let x = self.transf.ln_f.forward(x);
        self.lm_head.forward(x)
    }

    pub fn generate(&self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: Option<f64>,
        top_k: Option<usize>,) -> Tensor<B, 2, Int> {
        let mut idx_gen = idx.clone();
        for _ in 0..max_new_tokens {
            let idx_cond = if idx.dims()[1] <= self.block_size {
                idx.clone()
            } else {
                let start_idx = idx.dims()[1] - self.block_size;
                let end_idx = idx.dims()[1];
                idx.clone().slice([0..idx.dims()[0], start_idx..end_idx])
            };

            let temp = temperature.unwrap_or(1.0);

            let logits = self.forward(idx_cond);
            let logits = logits.clone().slice([
                0..logits.dims()[0],
                (logits.dims()[1] - 1)..logits.dims()[1],
                0..logits.dims()[2],
            ]);

            /* if top_k != None {
                let v = logits.topk(top_k.unwrap(), 1);
                let logits = logits.iter_dim(1);
                logits
            } */

            let probs = softmax(logits, 2);
            let idx_next = ops::multinominal_sample(probs.squeeze(0), 1);
            idx_gen = Tensor::cat(vec![idx.clone(), idx_next.int().unsqueeze()], 1);
        }
        idx_gen
    }
}
