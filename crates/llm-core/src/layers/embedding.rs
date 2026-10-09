use candle_core::{DType, Tensor};
use llm_types::Result;

use crate::Module;
pub struct VocabeEmbedding {
    weight: Tensor,
}

impl Module for VocabeEmbedding {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, _ctx: &mut crate::ForwardContext) -> Result<Tensor> {
        assert!(input.dtype() == DType::U32);
        let output = self.weight.index_select(input, 0)?;
        Ok(output)
    }
}
