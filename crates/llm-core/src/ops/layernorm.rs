use crate::impl_base_op;
use crate::ops::baseops::BaseOp;
use crate::ops::rms_norm;
use candle_core::Tensor;
use llm_types::Result;

pub trait LayerNorm: BaseOp {
    type Input;
    type Output;
    fn forward(&self, input: &Self::Input) -> Result<Self::Output>;
}
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl_base_op!(RMSNorm);

impl LayerNorm for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = rms_norm(input, &self.weight, self.eps)?;
        Ok(output)
    }
}
