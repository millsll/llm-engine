use candle_core::{D::Minus1, Tensor};
use llm_types::Result;

use crate::impl_base_op;
use crate::ops::baseops::BaseOp;
use crate::ops::baseops::silu_and_mul;

pub trait SiluAndMulImpl: BaseOp {
    type Input<'a>;
    type Output;
    fn forward<'a>(&self, x: Self::Input<'a>) -> Result<Self::Output>;
}

pub struct SiluAndMul;
impl_base_op!(SiluAndMul);

impl SiluAndMulImpl for SiluAndMul {
    type Input<'a> = &'a Tensor;
    type Output = Tensor;
    fn forward<'a>(&self, x: &'a Tensor) -> Result<Tensor> {
        let last_dim = x.dim(Minus1)?;
        let gate = x.narrow(Minus1, 0, last_dim / 2)?;
        let up = x.narrow(Minus1, last_dim / 2, last_dim / 2)?;
        let out = silu_and_mul(&gate, &up)?;
        Ok(out)
    }
}
