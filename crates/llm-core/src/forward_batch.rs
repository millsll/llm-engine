use candle_core::DType;
use candle_core::Device;
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

    pub total_query: usize,
}

impl ForwardBatch {
    pub fn new(
        input_ids: &[u32],
        positions: &[u32],
        query_lens: &[usize],
        prefix_lens: &[usize],
        device: &Device,
    ) -> Result<Self> {
        debug_assert!(query_lens.len() > 0);
        debug_assert!(prefix_lens.len() > 0);
        debug_assert!(input_ids.len() > 0);
        debug_assert!(positions.len() > 0);
        debug_assert!(query_lens.len() == prefix_lens.len());

        let input_ids = Tensor::from_slice(input_ids, input_ids.len(), device)?;
        let positions = Tensor::from_slice(positions, positions.len(), device)?;

        // query_lens usize->u32
        let query_lens: Vec<u32> = query_lens.iter().map(|&x| x as u32).collect();
        // 计算total query
        let total_query: usize = query_lens.iter().sum::<u32>() as usize;
        let prefix_lens: Vec<u32> = prefix_lens.iter().map(|&x| x as u32).collect();
        // seq_lens:
        let seq_lens: Vec<u32> = query_lens
            .iter()
            .zip(prefix_lens.iter())
            .map(|(x, y)| x + y)
            .collect();
        let max_query_len = *query_lens.iter().max().unwrap() as usize;
        let max_seq_len = *seq_lens.iter().max().unwrap() as usize;

        let mut acc = 0u32;
        let cu_seq_lens_q: Vec<u32> = query_lens
            .iter()
            .map(|&x| {
                acc += x;
                acc
            })
            .collect();
        acc = 0u32;
        let cu_seq_lens_kv: Vec<u32> = prefix_lens
            .iter()
            .map(|&x| {
                acc += x;
                x
            })
            .collect();
        // to device
        let query_lens = Tensor::from_slice(&query_lens, query_lens.len(), device)?;
        let prefix_lens = Tensor::from_slice(&prefix_lens, prefix_lens.len(), device)?;
        let seq_lens = Tensor::from_slice(&seq_lens, seq_lens.len(), device)?;
        let cu_seqlens_q = Tensor::from_slice(&cu_seq_lens_q, cu_seq_lens_q.len(), device)?;
        let cu_seqlens_kv = Tensor::from_slice(&cu_seq_lens_kv, cu_seq_lens_kv.len(), device)?;

        Ok(Self {
            input_ids,
            positions,
            query_lens,
            seq_lens,
            prefix_lens,
            max_query_len,
            max_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
            total_query,
        })
    }

    pub fn prefill_from_input_ids(input_ids: &[&[u32]], device: &Device) -> Result<Self> {
        //
        let query_lens: Vec<usize> = input_ids.iter().map(|row| row.len()).collect();
        let positions: Vec<u32> = query_lens.iter().flat_map(|&len| 0..len as u32).collect();
        let prefix_lens: Vec<usize> = vec![0usize; input_ids.len()];
        let input_ids: Vec<u32> = input_ids
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        Self::new(&input_ids, &positions, &query_lens, &prefix_lens, device)
    }

    pub fn select_last_query_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let indices = self
            .cu_seqlens_q
            .broadcast_sub(&Tensor::ones_like(&self.cu_seqlens_q)?)?;
        let max_index = indices.max_all()?.to_scalar::<u32>()?;
        if max_index as usize >= hidden_states.dim(0)? {
            return Err(llm_types::Error::InvalidRequest(
                "index select out of bound in forwardbatch".to_string(),
            ));
        }
        let output = hidden_states.index_select(&indices, 0)?;
        Ok(output)
    }
}
