use llm_types::Result;

use crate::forward_context::ForwardContext;
pub trait Module {
    type Input;
    type Output;

    fn forward(&self, input: &Self::Input, ctx: &mut ForwardContext) -> Result<Self::Output>;
}
