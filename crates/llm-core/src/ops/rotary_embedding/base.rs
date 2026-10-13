use std::sync::{Arc, Mutex, OnceLock};

use crate::{
    impl_base_op,
    ops::baseops::{BaseOp, apply_rotary},
};
use candle_core::{D::Minus1, DType, Device, DeviceLocation, Tensor, cuda::DeviceId};
use llm_types::{
    Result,
    model_config::{CoreConfig, LlamaConfig},
};
use std::collections::HashMap;

static ROPE_CACHE: OnceLock<Mutex<HashMap<RopeKey, Arc<RotaryEmbedding>>>> = OnceLock::new();

#[derive(Hash, Eq, PartialEq)]
pub struct RopeKey {
    head_dim: usize,
    max_position_embeddings: usize,
    /// true:f64 -> store:u64 support hash
    rope_theta: u64,
    dtype: DType,
    device: DeviceLocation,
}

impl RopeKey {
    pub fn from_config(config: &impl CoreConfig, dtype: DType, device: DeviceLocation) -> Self {
        Self {
            head_dim: config.get_head_dim(),
            max_position_embeddings: config.get_max_position_embeddings(),
            rope_theta: config.get_rope_theta().to_bits(),
            dtype,
            device,
        }
    }
}

pub fn get_or_create_rope(key: RopeKey, device: &Device) -> Result<Arc<RotaryEmbedding>> {
    let cache = ROPE_CACHE.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();

    if let Some(rope) = cache.get(&key) {
        return Ok(Arc::clone(rope));
    }

    let rope = Arc::new(RotaryEmbedding::new(&key, device)?);

    cache.insert(key, Arc::clone(&rope));

    Ok(rope)
}

pub trait RotaryEmbeddingImpl: BaseOp {
    type Input;
    type Output;
    fn forward(
        &self,
        positions: &Self::Input,
        query: &Self::Input,
        key: &Self::Input,
    ) -> Result<(Self::Output, Self::Output)>;
}

pub struct RotaryEmbedding {
    head_size: usize,
    rotary_dim: usize, // 0:rotary_dim rotate left passthrough
    cos_sin_cache: Tensor,
}

impl_base_op!(RotaryEmbedding);

impl RotaryEmbeddingImpl for RotaryEmbedding {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // positions:[batch,l]
        let positions = positions.flatten(Minus1, Minus1)?;
        let num_tokens = positions.shape().dim(Minus1)?;
        let cache_dim = self.cos_sin_cache.shape().dim(Minus1)?;
        let rotary_dim = cache_dim / 2;

        let cos = self.cos_sin_cache.narrow(Minus1, 0, rotary_dim)?;
        let cos = cos.index_select(&positions, 0)?;
        let sin = self.cos_sin_cache.narrow(Minus1, rotary_dim, rotary_dim)?;
        let sin = sin.index_select(&positions, 0)?;

        let q_shape = query.shape().clone();
        let q_heads = q_shape.dim(1)? / self.head_size;
        let query = query.reshape((num_tokens, q_heads, self.head_size))?;
        let query_rot = query.narrow(Minus1, 0, self.rotary_dim)?;
        let query_pass = query.narrow(Minus1, self.rotary_dim, self.head_size)?;
        let query_rot = apply_rotary(&query_rot, &cos, &sin)?;
        let query = Tensor::cat(&[query_rot, query_pass], Minus1)?.reshape(q_shape)?;

        let k_shape = key.shape().clone();
        let kv_heads = k_shape.dim(1)? / self.head_size;
        let key = key.reshape((num_tokens, kv_heads, self.head_size))?;
        let key_rot = key.narrow(Minus1, 0, self.rotary_dim)?;
        let key_pass = key.narrow(Minus1, self.rotary_dim, self.head_size)?;
        let key_rot = apply_rotary(&key_rot, &cos, &sin)?;
        let key = Tensor::cat(&[key_rot, key_pass], Minus1)?.reshape(k_shape)?;

        Ok((query, key))
    }
}

impl RotaryEmbedding {
    pub fn new(key: &RopeKey, device: &Device) -> Result<Self> {
        let head_dim = key.head_dim;
        let max_seq_len = key.max_position_embeddings;
        let theta = f64::from_bits(key.rope_theta);

        assert!(
            head_dim % 2 == 0,
            "RoPE head_dim mast be even,got {head_dim}"
        );

        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| {
                let exponext = 2.0 * i as f64 / head_dim as f64;
                (1.0 / theta.powf(exponext)) as f32
            })
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        let positions = Tensor::new(positions.as_slice(), device)?;

        let positions = positions.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;
        let freqs = positions.broadcast_mul(&inv_freq)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self {
            head_size: head_dim,
            rotary_dim: head_dim,
            cos_sin_cache: Tensor::cat(&[cos, sin], Minus1)?,
        })
    }
}
