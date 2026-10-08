use crate::{Module, forward_context::ForwardContext};
use candle_core::Tensor;
use llm_types::Result;
pub struct LinearLayer {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Module for LinearLayer {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, _ctx: &mut ForwardContext) -> Result<Tensor> {
        // 转置乘加
        let wt = self.weight.t()?;
        let output = input.matmul(&wt)?;
        match &self.bias {
            Some(bias) => Ok((output + bias)?),
            None => Ok(output),
        }
    }
}
