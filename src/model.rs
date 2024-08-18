use burn::{module::Module, prelude::*, nn::attention};
use nn::{LayerNormConfig, LinearConfig};

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
    n_head: usize,
    n_embd: usize,
    bias: bool,
    dropout: f64,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.n_embd % self.n_head == 0);
        SelfAttention {
            c_attn: nn::LinearConfig::new(self.n_embd, self.n_embd * 3).with_bias(self.bias).init(device),
            c_proj: nn::LinearConfig::new(self.n_embd, self.n_embd).with_bias(self.bias).init(device),
            resid_dropout: nn::DropoutConfig::new(self.dropout).init(),
            attn: attention::MultiHeadAttentionConfig::new(self.n_embd, self.n_head).with_dropout(self.dropout).init(device),
            n_embd: self.n_embd,
            n_head: self.n_head
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
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            ln_1: nn::LayerNormConfig::new(self.attn_config.n_embd).init(device),
            attn: self.attn_config.init(device),
            ln_2: LayerNormConfig::new(self.attn_config.n_embd).init(device),
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