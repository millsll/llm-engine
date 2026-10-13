use crate::{Module, forward_context::ForwardContext};
use candle_core::Tensor;
use candle_nn::VarBuilder;
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

impl LinearLayer {
    pub fn load(vb: &VarBuilder, shape: &[usize]) -> Result<Self> {
        let weight = vb.get(shape, "weight")?;
        Ok(Self { weight, bias: None })
    }

    pub fn load_fuse_weight(vb: &VarBuilder, shapes: &[&[usize]], names: &[&str]) -> Result<Self> {
        let weights = names
            .into_iter()
            .zip(shapes.into_iter())
            .map(|(name, &shape)| vb.get(shape, name).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self { weight, bias: None })
    }
}
