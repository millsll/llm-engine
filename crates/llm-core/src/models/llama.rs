//! Llama 3.x 模型。
//!
//! 与 Qwen2 的结构差异：QKV 是否带 bias、RoPE scaling 方式、是否 tie embedding。
//!
//! TODO(phase-1): 在 qwen2 数值对齐通过后实现。

use crate::{
    ForwardContext,
    layers::{
        attention::{Attention, AttentionInput},
        linear::LinearLayer,
    },
    ops::{
        activation::{self, SiluAndMulImpl},
        layernorm::{LayerNorm, RMSNorm},
        rms_norm,
        rotary_embedding::base::{RotaryEmbedding, RotaryEmbeddingImpl},
    },
};
use candle_core::{D::Minus1, Tensor};
use llm_types::Result;

use crate::Module;

pub struct LlamaMLP {
    gate_up_proj: LinearLayer,
    down_proj: LinearLayer,
    hidden_act: activation::SiluAndMul,
}

impl Module for LlamaMLP {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, ctx: &mut ForwardContext) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(input, ctx)?;
        let x = self.hidden_act.forward(&gate_up)?;
        let x = self.down_proj.forward(&x, ctx)?;
        Ok(x)
    }
}

pub struct LlamaAttention {
    qkv_proj: LinearLayer,
    o_oproj: LinearLayer,

    // rotary
    rotary_embed: RotaryEmbedding,
    // attention size
    q_size: usize,
    kv_size: usize,

    attn: Attention,
}

impl Module for LlamaAttention {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Self::Input, ctx: &mut ForwardContext) -> Result<Self::Output> {
        // 计算qkv 拼起来一次算完
        let qkv = self.qkv_proj.forward(input, ctx)?;
        // 拆分qkv total h
        let q = qkv.narrow(Minus1, 0, self.q_size)?;
        let k = qkv.narrow(Minus1, self.q_size, self.q_size + self.kv_size)?;
        let v = qkv.narrow(
            Minus1,
            self.q_size + self.kv_size,
            self.q_size + 2 * self.kv_size,
        )?;
        let (q, k) = self.rotary_embed.forward(&ctx.batch.positions, &q, &k)?;
        let attn_output = self.attn.forward(&AttentionInput::new(&q, &k, &v), ctx)?;
        let output = self.o_oproj.forward(&attn_output, ctx)?;
        Ok(output)
    }
}

pub struct LlamaDecoderLayer {
    attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attn_layernorm: RMSNorm,
}

impl Module for LlamaDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, ctx: &mut ForwardContext) -> Result<Tensor> {
        let residual = input.clone();
        let hidden_states = self.input_layernorm.forward(input)?;
        let hidden_states = self.attn.forward(&hidden_states, ctx)?;
        let hidden_states = hidden_states.add(&residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attn_layernorm.forward(input)?;
        let hidden_states = self.mlp.forward(&hidden_states, ctx)?;
        let output = hidden_states.add(&residual)?;
        Ok(output)
    }
}
