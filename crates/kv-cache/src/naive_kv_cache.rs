use candle_core::{DType, Device, Tensor};
use llm_types::{KvCache, Result};
use std::collections::HashMap;
pub struct NaiveKvCache {
    // layer -> [kvs...]->[total_tokens,hkv,head_dim]
    k_pool: Vec<Tensor>,
    v_pool: Vec<Tensor>,

    seq_to_slots: HashMap<SeqId, BatchSlot>,
    slots: Vec<SeqSlot>,
    // 空闲slots 线性分配不回收
    next_free_slot: usize,
    max_total_tokens: usize,
    // metadata
    dtype: DType,
    device: Device,
}

struct SeqSlot {
    start: usize,
    len: usize,
    max_len: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BatchSlot(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SeqId(pub usize);

impl NaiveKvCache {
    pub fn new(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_tokens: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut k_pool = Vec::with_capacity(num_layers);
        let mut v_pool = Vec::with_capacity(num_layers);
        let shape = (max_tokens, n_kv_heads, head_dim);
        for i in 0..num_layers {
            k_pool.push(Tensor::zeros(shape, dtype, device)?);
            v_pool.push(Tensor::zeros(shape, dtype, device)?);
        }
        let seq_slots = HashMap::<SeqId, SeqSlot>::new();
        Ok(Self {
            k_pool,
            v_pool,
            seq_slots,
            next_free_slot: 0,
            max_total_tokens: max_tokens,
            dtype: dtype,
            device: device.clone(),
        })
    }
}

impl KvCache for NaiveKvCache {
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {}

    fn get(&self, layer: usize) -> Result<(Tensor, Tensor)> {}

    fn reset(&mut self) -> Result<()> {}

    fn seq_len(&self) -> usize {}
}
