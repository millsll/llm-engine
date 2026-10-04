use llm_types::KvCache;
pub struct ForwardContext<'a, C: KvCache> {
    pub kv_cache: &'a mut C,
}
