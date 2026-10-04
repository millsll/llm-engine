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
