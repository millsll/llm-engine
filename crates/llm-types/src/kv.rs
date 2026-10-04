//! KV cache 的契约。
//!
//! 契约放在这里而不是 `llm-core` 或 `mem-cache`，是为了让两边都不必依赖对方：
//! 前向计算只需要 [`KvCache`]，实现在 `mem-cache`。

use candle_core::Tensor;

use crate::error::Result;

/// 单条序列的 KV cache。
///
/// ## 形状约定
///
/// 张量布局与 candle 一致，`head_dim` 在最后一维：
///
/// - [`KvCache::append`] 传入的 `k` / `v`：`(num_kv_heads, n_new_tokens, head_dim)`
/// - [`KvCache::get`] 返回的 `k` / `v`：`(num_kv_heads, total_tokens, head_dim)`
///
/// 这里刻意不暴露 batch 维：每条序列持有自己的 cache 对象，批量推理由调度层
/// 组装。分页实现在满足同一语义的前提下，把数据散落在物理块中。
///
/// ## 已知的性能取舍
///
/// [`KvCache::get`] 返回连续张量，分页实现需要做一次 gather。生产路径上这一步
/// 会由自定义 paged-attention 算子绕过——届时由调度层把
/// [`BlockTable`](crate::block::BlockTable) 作为前向的数据参数传入，
/// 而不是让前向去反向依赖存储层。
/// 现在不加，是为了让 Phase 1 的接口保持最小。
pub trait KvCache {
    /// 追加某一层新算出的 K / V。
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()>;

    /// 取出某一层截止到当前的完整 K / V。
    fn get(&self, layer: usize) -> Result<(Tensor, Tensor)>;

    /// 当前已缓存的 token 数。
    fn seq_len(&self) -> usize;

    /// 清空全部缓存。序列复用同一个 cache 对象时调用。
    fn reset(&mut self) -> Result<()>;
}
