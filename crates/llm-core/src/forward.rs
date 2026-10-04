use llm_types::KvCache;

use crate::context::ForwardContext;

pub trait Forward {
    type Input;
    type Output;
    type Error;

    fn forward<C: KvCache>(
        &self,
        input: Self::Input,
        ctx: &mut ForwardContext<C>,
    ) -> Result<Self::Output, Self::Error>;
}
