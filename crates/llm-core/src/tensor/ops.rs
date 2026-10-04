//! 自研算子。
//!
//! 规划中的五个基础算子：
//!
//! - [`apply_rotary`] —— RoPE（本次实现）
//! - `rms_norm(x, weight, eps)` —— RMSNorm
//! - `softmax_last_dim(x)` —— 末维 softmax
//! - `silu_and_mul(gate, up)` —— SwiGLU 激活
//! - `repeat_kv(kv, n_rep)` —— GQA 的 KV 复制
//!
//! 形状一律不带 batch 维，完整规格见 `docs/tensor-ops.md`。
//!
//! TODO(phase-1): 实现其余四个算子。

use candle_core::{D, Tensor};

use llm_types::Result;

/// 应用 RoPE，Llama / Qwen 使用的 `rotate_half` 形式。
///
/// # 形状
///
/// - `x`：`(n_head, seq_len, head_dim)`
/// - `cos` / `sin`：`(seq_len, head_dim)`，**整头宽**——也就是 HuggingFace
///   里 `emb = cat(freqs, freqs)` 之后的结果，不是半宽
/// - 返回：`(n_head, seq_len, head_dim)`
///
/// `cos` / `sin` 靠广播对齐到 `n_head` 维。
///
/// # 语义
///
/// ```text
/// rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
/// out            = x * cos + rotate_half(x) * sin
/// ```
///
/// 展开后等价于把前后两半配对做二维旋转，每一对（`i` 取 `0..d/2`）：
///
/// ```text
/// out[i]       = x[i]       * cos[i] - x[i + d/2] * sin[i]
/// out[i + d/2] = x[i + d/2] * cos[i] + x[i]       * sin[i]
/// ```
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rot = rotate_half(x)?;
    let out = (x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?)?;
    Ok(out)
}

/// `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`。
///
/// candle-nn 内部有一份同名的私有实现（`rotary_emb.rs`），语义一致；
/// 复制一份是因为它没有公开。
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, last_dim - half)?;
    Ok(Tensor::cat(&[&second.neg()?, &first], D::Minus1)?)
}

/// silu(gate)*up
pub fn silu_and_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let silu_gate = candle_nn::ops::silu(gate)?;
    let out = (silu_gate * up)?;
    Ok(out)
}

pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let out = candle_nn::ops::softmax_last_dim(x)?;
    Ok(out)
}

/// x[B,Hn,L,d]->x[B,Hn,n_rep,L,d]->[B,Hq,L,d]
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.contiguous()?);
    }
    let (batch, n_kv, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((batch, n_kv, n_rep, seq_len, head_dim))?
        .reshape((batch, n_kv * n_rep, seq_len, head_dim))?;
    Ok(x)
}

/// rmsnorm
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let out = candle_nn::ops::rms_norm(x, weight, eps)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// 两个张量逐元素差的最大绝对值。
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let diff = (a - b)?.abs()?;
        Ok(diff.max_all()?.to_scalar::<f32>()?)
    }

    /// 小输入对拍 `candle_nn::rotary_emb::rope`，同时钉住手算结果。
    ///
    /// 两边的接口有三处差别，测试里逐个弥合：
    ///
    /// - 形状：本函数收 `(n_head, S, d)`，candle 要 4D `(batch, n_head, S, d)`
    /// - cos/sin 宽度：本函数收 HF 的整头宽 `(S, d)`，
    ///   candle 收半宽 `(S, d/2)`（它在内部 `cat(&[cos, cos], -1)` 补回整宽）
    /// - 连续性：candle 要求三者都 contiguous，`narrow` 出来的两半不是，
    ///   所以要显式 `contiguous()`
    #[test]
    fn apply_rotary_matches_candle_nn_rope() -> Result<()> {
        let dev = Device::Cpu;

        // (n_head, S, d) = (2, 3, 4)，第二个头是第一个头的取负，
        // 这样旋转的线性性可以肉眼核对。
        let x = Tensor::new(
            &[
                [
                    [1.0f32, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [-1.0f32, -2.0, -3.0, -4.0],
                    [-5.0, -6.0, -7.0, -8.0],
                    [-9.0, -10.0, -11.0, -12.0],
                ],
            ],
            &dev,
        )?;

        // 整头宽 cos/sin，(S, d) = (3, 4)。取值都是 0 / 1 / -1，
        // f32 下精确可表示，所以后面的期望结果是精确值而不是近似值。
        //
        // 每行满足 HF 的 emb = cat(freqs, freqs) 结构，即 row[i] == row[i + d/2]。
        let cos = Tensor::new(
            &[
                [1.0f32, 1.0, 1.0, 1.0],   // pos 0: 全 1，恒等
                [1.0f32, 0.0, 1.0, 0.0],   // pos 1: 第 1 对转 90 度
                [0.0f32, -1.0, 0.0, -1.0], // pos 2: 第 0 对转 90 度
            ],
            &dev,
        )?;
        let sin = Tensor::new(
            &[
                [0.0f32, 0.0, 0.0, 0.0],
                [0.0f32, 1.0, 0.0, 1.0],
                [1.0f32, 0.0, 1.0, 0.0],
            ],
            &dev,
        )?;

        let ours = apply_rotary(&x, &cos, &sin)?;

        // --- 1. 手算期望 ---
        //
        // head 0 逐位置推导（c = cos[i], s = sin[i]）：
        //   pos 0  c=[1,1], s=[0,0]  -> [1, 2, 3, 4]（恒等）
        //   pos 1  c=[1,0], s=[0,1]  -> [5, -8, 7, 6]
        //             out[0]=5*1-7*0=5   out[2]=7*1+5*0=7
        //             out[1]=6*0-8*1=-8  out[3]=8*0+6*1=6
        //   pos 2  c=[0,-1], s=[1,0] -> [-11, -10, 9, -12]
        //             out[0]=9*0-11*1=-11  out[2]=11*0+9*1=9
        //             out[1]=10*(-1)-12*0=-10  out[3]=12*(-1)+10*0=-12
        //
        // head 1 是 head 0 的取负（旋转是线性的）。
        let expected = Tensor::new(
            &[
                [
                    [1.0f32, 2.0, 3.0, 4.0],
                    [5.0, -8.0, 7.0, 6.0],
                    [-11.0, -10.0, 9.0, -12.0],
                ],
                [
                    [-1.0f32, -2.0, -3.0, -4.0],
                    [-5.0, 8.0, -7.0, -6.0],
                    [11.0, 10.0, -9.0, 12.0],
                ],
            ],
            &dev,
        )?;

        let vs_hand = max_abs_diff(&ours, &expected)?;
        println!("max |ours - 手算| = {vs_hand:e}");
        assert_eq!(vs_hand, 0.0, "与手算结果不一致，差 {vs_hand}");

        // --- 2. 对拍 candle_nn::rope ---

        // candle 收半宽 cos/sin。narrow 出来的结果不连续，必须 contiguous()。
        let cos_half = cos.narrow(D::Minus1, 0, 2)?.contiguous()?;
        let sin_half = sin.narrow(D::Minus1, 0, 2)?.contiguous()?;

        // candle 要 4D，补一个 batch 维再 squeeze 回来。
        let theirs =
            candle_nn::rotary_emb::rope(&x.unsqueeze(0)?, &cos_half, &sin_half)?.squeeze(0)?;

        assert_eq!(ours.dims(), theirs.dims(), "形状不一致");

        let vs_candle = max_abs_diff(&ours, &theirs)?;
        println!("max |ours - candle_nn::rope| = {vs_candle:e}");
        assert!(
            vs_candle < 1e-6,
            "与 candle_nn::rope 差 {vs_candle}，超出容差"
        );

        // --- 3. 顺带确认没写成交错式 ---
        //
        // rope_i 是另一种配对方式（相邻两维一组），形状一样但结果不同。
        // 如果这里意外相等，说明测试输入选得没有区分度，得换一组。
        let interleaved =
            candle_nn::rotary_emb::rope_i(&x.unsqueeze(0)?, &cos_half, &sin_half)?.squeeze(0)?;
        let vs_rope_i = max_abs_diff(&ours, &interleaved)?;
        println!("max |ours - rope_i| = {vs_rope_i:e}  (应当明显大于 0)");
        assert!(
            vs_rope_i > 1e-3,
            "与交错式的差只有 {vs_rope_i}，这组测试数据区分不出两种配对方式"
        );

        Ok(())
    }
}
