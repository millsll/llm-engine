use candle_core::Tensor;

use crate::Module;

pub struct LogitsProcess {}

impl Module for LogitsProcess {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(
        &self,
        input: &Self::Input,
        ctx: &mut crate::ForwardContext,
    ) -> llm_types::Result<Self::Output> {
        Ok(input.clone())
    }
}
