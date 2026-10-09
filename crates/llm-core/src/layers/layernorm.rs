use candle_core::Tensor;
use llm_types::Result;

use crate::{Module, ops::rms_norm};

pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl Module for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, _ctx: &mut crate::ForwardContext) -> Result<Tensor> {
        let output = rms_norm(input, &self.weight, self.eps)?;
        Ok(output)
    }
}
