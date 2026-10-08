use crate::ForwardContext;
use crate::Module;
use candle_core::Tensor;
use candle_nn::varlen_attention::flash_attn_varlen_unfused;
use llm_types::Result;
pub struct Attention {
    head_dim: usize,
    num_heads: usize,
}

pub struct AttentionInput {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}
impl AttentionInput {
    pub fn new(q: &Tensor, k: &Tensor, v: &Tensor) -> Self {
        Self {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
        }
    }
}

impl Module for Attention {
    type Input = AttentionInput;
    type Output = Tensor;
    fn forward(&self, input: &AttentionInput, ctx: &mut ForwardContext) -> Result<Tensor> {
        // q: total, h*head_dim
        let (total_q, _) = input.q.dims2()?;
        let (total_kv, kv_hidden) = input.k.dims2()?;
        let q = input.q.reshape((total_q, self.num_heads, self.head_dim))?;
        let k = input
            .k
            .reshape((total_kv, kv_hidden / self.head_dim, self.head_dim))?;
        let v = input
            .v
            .reshape((total_kv, kv_hidden / self.head_dim, self.head_dim))?;
        // total h d
        let output = flash_attn_varlen_unfused(
            &q,
            &k,
            &v,
            None,
            &ctx.batch.query_lens,
            &ctx.batch.seq_lens,
            ctx.batch.max_query_len,
            ctx.batch.max_seq_len,
            (1.0 / (self.head_dim as f64).sqrt()) as f32,
            true,
            None,
            None,
        )?;
        Ok(output)
    }
}
