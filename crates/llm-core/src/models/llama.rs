//! Llama 3.x 模型。
//!
//! 与 Qwen2 的结构差异：QKV 是否带 bias、RoPE scaling 方式、是否 tie embedding。
//!
//! TODO(phase-1): 在 qwen2 数值对齐通过后实现。

use std::sync::Arc;

use crate::layers::layernorm::RMSNorm;
use crate::models::Model;
use crate::ops::rotary_embedding::base::{RopeKey, get_or_create_rope};
use crate::{
    ForwardContext,
    layers::{
        attention::{Attention, AttentionInput},
        embedding::VocabeEmbedding,
        linear::LinearLayer,
    },
    ops::{
        activation::{self, SiluAndMulImpl},
        rotary_embedding::base::{RotaryEmbedding, RotaryEmbeddingImpl},
    },
};
use candle_core::{D::Minus1, Tensor};
use candle_nn::{VarBuilder, rotary_emb};
use llm_types::model_config::CoreConfig;
use llm_types::model_config::LlamaConfig;
use llm_types::{Result, device};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

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

impl LlamaMLP {
    pub fn load(vb: &VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let vb_mlp = vb.clone().pp("mlp");
        // load gate up
        let gate_up_proj = LinearLayer::load_fuse_weight(
            &vb_mlp,
            &[&[config.intermediate_size, config.hidden_size] as &[usize]; 2],
            &["gate_proj.weight", "up_proj.weight"],
        )?;
        // loaddown
        let down_proj = LinearLayer::load(
            &vb_mlp.clone().pp("down_proj"),
            &[config.hidden_size, config.intermediate_size],
        )?;
        let hidden_act = activation::SiluAndMul {};
        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_act,
        })
    }
}

pub struct LlamaAttention {
    qkv_proj: LinearLayer,
    o_proj: LinearLayer,

    // rotary
    rotary_embed: Arc<RotaryEmbedding>,
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
        let k = qkv.narrow(Minus1, self.q_size, self.kv_size)?;
        let v = qkv.narrow(Minus1, self.q_size + self.kv_size, self.kv_size)?;
        let (q, k) = self.rotary_embed.forward(&ctx.batch.positions, &q, &k)?;
        let attn_output = self.attn.forward(&AttentionInput::new(&q, &k, &v), ctx)?;
        let output = self.o_proj.forward(&attn_output, ctx)?;
        Ok(output)
    }
}

impl LlamaAttention {
    pub fn load(vb: &VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let vb_attn = vb.clone().pp("self_attn");
        let o_proj = LinearLayer::load(
            &vb_attn.clone().pp("o_proj"),
            &[
                config.head_dim * config.num_attention_heads,
                config.hidden_size,
            ],
        )?;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_key_value_heads * config.head_dim;
        let qkv_proj = LinearLayer::load_fuse_weight(
            &vb_attn,
            &[
                &[q_size, config.hidden_size],
                &[kv_size, config.hidden_size],
                &[kv_size, config.hidden_size],
            ],
            &["q_proj.weight", "k_proj.weight", "v_proj.weight"],
        )?;
        let attn = Attention {
            head_dim: config.head_dim,
            num_heads: config.num_attention_heads,
        };
        let rotary_key =
            RopeKey::from_config(config, candle_core::DType::F32, vb.device().location());
        let rotary_embed = get_or_create_rope(rotary_key, vb.device())?;
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_embed,
            q_size,
            kv_size,
            attn,
        })
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
        let hidden_states = self.input_layernorm.forward(input, ctx)?;
        let hidden_states = self.attn.forward(&hidden_states, ctx)?;
        let hidden_states = hidden_states.add(&residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attn_layernorm.forward(&hidden_states, ctx)?;
        let hidden_states = self.mlp.forward(&hidden_states, ctx)?;
        let output = hidden_states.add(&residual)?;
        Ok(output)
    }
}

impl LlamaDecoderLayer {
    pub fn load(vb: &VarBuilder, config: &LlamaConfig, layer_id: usize) -> Result<Self> {
        let vb_layer = vb.clone().pp(&layer_id.to_string());
        // 加载inputlayer norm
        let vb_input_layernorm = vb_layer.clone().pp("input_layernorm");
        let input_layernorm = RMSNorm::load(
            &vb_input_layernorm,
            config.hidden_size,
            config.rms_norm_eps as f32,
        )?;
        let vb_post_attn_layernorm = vb_layer.clone().pp("post_attention_layernorm");
        let post_attn_layernorm = RMSNorm::load(
            &vb_post_attn_layernorm,
            config.hidden_size,
            config.rms_norm_eps as f32,
        )?;
        let mlp = LlamaMLP::load(&vb_layer, config)?;
        let attn = LlamaAttention::load(&vb_layer, config)?;
        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attn_layernorm,
        })
    }
}

pub struct LlamaModel {
    embed_tokens: VocabeEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RMSNorm,
}

pub struct LlamaModelInput {
    pub(crate) input: Tensor,
    pub(crate) input_embedding: Option<Tensor>,
}

impl Module for LlamaModel {
    type Input = LlamaModelInput;
    type Output = Tensor;
    fn forward(&self, input: &LlamaModelInput, ctx: &mut ForwardContext) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(&input.input, ctx)?;
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, ctx)?;
        }
        let output = self.norm.forward(&hidden_states, ctx)?;

        Ok(output)
    }
}

impl LlamaModel {
    pub fn load(vb: &VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let vb_model_layer = vb.clone().pp("model.layers");
        //let embed_tokens
        // 并发加载layers
        let layers = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|i| LlamaDecoderLayer::load(&vb_model_layer, config, i))
            .collect::<Result<Vec<_>>>()?;
        // 加载 norm
        let vb_norm = vb.clone().pp("model.norm");
        let norm = RMSNorm::load(&vb_norm, config.hidden_size, config.rms_norm_eps as f32)?;
        // 加载 embedding
        let vb_vocabe = vb.clone().pp("model.embed_tokens");
        let embed_tokens =
            VocabeEmbedding::load(&vb_vocabe, &[config.vocab_size, config.hidden_size])?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

pub struct LlamaForCausalLM {
    model: LlamaModel,
    lm_head: LinearLayer,
}

impl Module for LlamaForCausalLM {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Self::Input, ctx: &mut ForwardContext) -> Result<Self::Output> {
        let hidden_states = self.model.forward(
            &LlamaModelInput {
                input: input.clone(),
                input_embedding: None,
            },
            ctx,
        )?;
        let last_hidden_states = ctx.batch.select_last_query_tokens(&hidden_states)?;
        let logits = self.lm_head.forward(&last_hidden_states, ctx)?;
        Ok(logits)
    }
}

impl Model for LlamaForCausalLM {}

impl LlamaForCausalLM {
    pub fn load(vb: &VarBuilder, config: &LlamaConfig) -> Result<Self> {
        // 根 load一下lmhead
        let vb_bf16 = vb.clone().set_dtype(candle_core::DType::BF16);
        // pp 只到模块名，叶子名 "weight" 由 VocabeEmbedding::load 自己拼。
        // 写成 pp("lm_head.weight") 会去找 "lm_head.weight.weight"。
        let vb_lm_head = vb_bf16.clone().pp("lm_head");
        let lm_head = LinearLayer::load(&vb_lm_head, &[config.vocab_size, config.hidden_size])?;
        //let lm_head = vb.get((config.vocab_size, config.hidden_size), "lm_head.weight")?;
        let model = LlamaModel::load(&vb_bf16.clone(), config)?;
        Ok(Self { model, lm_head })
    }
}
