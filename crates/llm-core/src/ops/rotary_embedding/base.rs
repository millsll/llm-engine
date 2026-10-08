use crate::{
    impl_base_op,
    ops::baseops::{BaseOp, apply_rotary},
};
use candle_core::{D::Minus1, Tensor};
use llm_types::Result;

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

        let cos = self
            .cos_sin_cache
            .narrow(Minus1, 0, self.cos_sin_cache.elem_count() / 2)?;
        let cos = cos.index_select(&positions, 0)?;
        let sin = self.cos_sin_cache.narrow(
            Minus1,
            self.cos_sin_cache.elem_count() / 2,
            self.cos_sin_cache.elem_count() / 2,
        )?;
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
