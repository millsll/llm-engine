use candle_core::DType;
use candle_core::Tensor;
use llm_types::Result;

pub struct ForwardBatch {
    pub input_ids: Tensor,
    pub positions: Tensor,
    // 本次forward的querylen
    pub query_lens: Tensor,
    // forward后的总seqlen
    pub seq_lens: Tensor,
    // forward前已有的kvcache长度
    pub prefix_lens: Tensor,

    pub max_query_len: usize,
    pub max_seq_len: usize,

    pub cu_seqlens_q: Tensor,
    pub cu_seqlens_kv: Tensor,
}

impl ForwardBatch {
    pub fn new(
        input_ids: &Tensor,
        positions: &Tensor,
        query_lens: &Tensor,
        prefix_lens: &Tensor,
    ) -> Result<Self> {
        debug_assert!(input_ids.dtype() == DType::U32);
        debug_assert!(positions.dtype() == DType::U32);
        debug_assert!(query_lens.dtype() == DType::U32);
        debug_assert!(prefix_lens.dtype() == DType::U32);
        let seq_lens = (query_lens + prefix_lens)?;
        let cu_seqlens_q = query_lens.cumsum(0)?;
        let cu_seqlens_kv = prefix_lens.cumsum(0)?;
        let max_query_len = query_lens.max(0)?.to_scalar::<u32>()? as usize;
        let max_seq_len = seq_lens.max(0)?.to_scalar::<u32>()? as usize;
        Ok(Self {
            input_ids: input_ids.clone(),
            positions: positions.clone(),
            query_lens: query_lens.clone(),
            seq_lens,
            prefix_lens: prefix_lens.clone(),
            max_query_len,
            max_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
        })
    }

    pub fn select_last_query_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let indices = self
            .cu_seqlens_q
            .broadcast_sub(&Tensor::ones_like(&self.cu_seqlens_q)?)?;
        let output = hidden_states.index_select(&indices, 0)?;
        Ok(output)
    }
}
