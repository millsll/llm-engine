use candle_core::Tensor;
use candle_nn::VarBuilder;
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

impl RMSNorm {
    pub fn load(vb: &VarBuilder, norm_size: usize, eps: f32) -> Result<Self> {
        let weight = vb.get(norm_size, "weight")?;
        let eps = eps;

        Ok(Self { weight, eps })
    }
}
