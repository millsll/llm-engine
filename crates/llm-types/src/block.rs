//! 物理块相关的纯数据类型。
//!
//! 放在 `llm-types` 而不是 `mem-cache`，是为了让块表既能被 paged-attention
//! 算子消费，也能被调度层构造，而三个 crate 之间不需要互相依赖。

/// 物理块编号，索引显存块池中的槽位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// 取原始编号。
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// 一条序列的「逻辑块 -> 物理块」映射。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
}

impl BlockTable {
    /// 构造空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一个物理块。
    pub fn push(&mut self, id: BlockId) {
        self.blocks.push(id);
    }

    /// 按逻辑下标取物理块。
    pub fn get(&self, logical: usize) -> Option<BlockId> {
        self.blocks.get(logical).copied()
    }

    /// 逻辑块数量。
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// 是否没有任何块。
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// 全部物理块。
    pub fn as_slice(&self) -> &[BlockId] {
        &self.blocks
    }

    /// 清空。真正的释放语义由调用方负责。
    pub fn clear(&mut self) {
        self.blocks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_table_indexes_logically() {
        let mut table = BlockTable::new();
        assert!(table.is_empty());
        table.push(BlockId(7));
        table.push(BlockId(3));
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(0), Some(BlockId(7)));
        assert_eq!(table.get(1), Some(BlockId(3)));
        assert_eq!(table.get(2), None);
        assert_eq!(table.as_slice()[1].raw(), 3);
        table.clear();
        assert!(table.is_empty());
    }
}
