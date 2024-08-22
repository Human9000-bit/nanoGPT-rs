use burn::{nn::attention, prelude::*};
use nn::{LayerNormConfig, LinearConfig};

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
    config: ModelConfig
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.config.n_embd % self.config.n_head == 0);
        SelfAttention {
            c_attn: nn::LinearConfig::new(self.config.n_embd, self.config.n_embd * 3).with_bias(self.config.bias).init(device),
            c_proj: nn::LinearConfig::new(self.config.n_embd, self.config.n_embd).with_bias(self.config.bias).init(device),
            resid_dropout: nn::DropoutConfig::new(self.config.dropout).init(),
            attn: attention::MultiHeadAttentionConfig::new(self.config.n_embd, self.config.n_head).with_dropout(self.config.dropout).init(device),
            n_embd: self.config.n_embd,
            n_head: self.config.n_head
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
    bias: bool
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            c_fc: nn::LinearConfig::new(self.n_embd, self.n_embd * 4).with_bias(self.bias).init(device),
            gelu: nn::Gelu::new(),
            c_proj: nn::LinearConfig::new(self.n_embd * 4, self.n_embd).with_bias(self.bias).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init()
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
    config: ModelConfig
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            ln_1: nn::LayerNormConfig::new(self.config.n_embd).init(device),
            attn: self.attn_config.init(device),
            ln_2: LayerNormConfig::new(self.config.n_embd).init(device),
            mlp: self.mlp_config.init(device)
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
    block_config: BlockConfig
}

impl TranformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let mut blocks = Vec::new();
        let (attn_config, mlp_config, model_config) = (self.block_config.attn_config.clone(), self.block_config.mlp_config.clone(), self.config.clone());
        for i in [0..self.config.n_layer] {
            blocks.push(BlockConfig::new(attn_config.clone(), mlp_config.clone(), self.config.clone()).init(device))
        }
        Transformer {
            wte: nn::EmbeddingConfig::new(self.config.vocab_size, self.config.n_embd).init(device),
            wpe: nn::EmbeddingConfig::new(self.config.block_size, self.config.n_embd).init(device),
            drop: nn::DropoutConfig::new(self.config.dropout).init(),
            h: blocks,
            ln_f: LayerNormConfig::new(self.config.n_embd).init(device)
        }
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    transf: Transformer<B>,
    lm_head: nn::Linear<B>,
}


#[derive(Config, Debug)]
pub struct GPTConfig {
    transf: TranformerConfig,
    block_conf: BlockConfig,
    config: ModelConfig
}

impl GPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT<B> {
        let transf = self.transf.clone().init(device);
        
        println!("Number of parameters: {}", &transf.num_params());
        GPT {
           transf,
           lm_head: LinearConfig::new(self.config.n_embd, self.config.vocab_size)
               .with_bias(self.config.bias).init(device)
        }
        
        
    }
}