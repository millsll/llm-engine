# tensor 模块执行流程

本文是 `llm-core::tensor` 的实现蓝图，覆盖模块划分、五个基础算子的执行流程、
精度策略与验证方法。供审查后再动手写代码。

绘图约定与 `forward-pass.md` 一致：**带右边框的方框内只写 ASCII**，
中日韩字符一律放在框外。

---

## 0. 范围

### 做什么

实现 `llm-core::tensor::ops` 里的五个基础算子，它们是整模型前向的全部非线性和
形状变换：`rms_norm`、`apply_rotary`、`silu_and_mul`、`softmax_last_dim`、
`repeat_kv`。

### 不做什么

- **不做 matmul、embedding、加法**。这些直接用 candle 的一等接口，
  不值得包一层。
- **不做 attention 的整体组装**。`softmax` 只是它的一步，causal mask 的构造
  和 `q·kT` 放在 `models::layers::Attention`，因为那里才知道 `S` 和 `T`。
- **不做算子融合**。Phase 5 再考虑把 `rms_norm` 和 `silu_and_mul` 融进 kernel。

### 与 candle-nn 的关系

candle-nn 提供了可对拍的等价实现，但**签名与语义并不一一对应**。
下表在 candle 0.11 的源码上逐个核对过，照着写可以避免踩空：

| 我们的算子 | candle-nn 的对应物 | 差异 |
| --- | --- | --- |
| `rms_norm` | `ops::rms_norm` | 它的 `eps` 是 `f32`，我们是 `f64`（来自 `ModelConfig`） |
| `softmax_last_dim` | `ops::softmax_last_dim` | 签名一致，可直接对拍 |
| `apply_rotary` | `rotary_emb::rope` | **必须用 `rope` 而不是 `rope_i`**，见 3.2 |
| `silu_and_mul` | `ops::swiglu` | 它的输入是拼接后的**单张量**，对拍前要先 `cat` |
| `repeat_kv` | 无 | candle 没有，参考实现要自己写 |

我们仍然自己写这五个算子，理由是 Phase 5 之后要换成自己的 kernel，
而测试里同时拿 candle-nn 和纯 Rust 参考实现对拍，等于免费多一个交叉验证。

---

## 1. 模块结构

```
   src/tensor/
     ├── mod.rs      统一 re-export candle 的 Tensor / Device / DType / IndexOp
     └── ops.rs      五个基础算子
```

`mod.rs` 是唯一的 candle 入口。除了它和 `ops.rs`，`llm-core` 里其他模块
**不应直接 `use candle_core::{...}`**——这条约束让将来替换张量层只需要改一处。

```
   models::layers
        │
        │  use crate::tensor::{Tensor, DType};     对
        │  use candle_core::Tensor;                错
        ▼
   tensor::ops  ──►  tensor::mod  ──►  candle
```

---

## 2. 通用约定

### 2.1 形状记法

| 符号 | 含义 | Qwen2.5-0.5B |
| --- | --- | --- |
| `S` | 本次前向的 token 数 | prefill 时 = prompt 长度，decode 时 = 1 |
| `T` | cache 中已有的 token 数 | prefill 时 0，decode 时逐步增长 |
| `H` | hidden size | 896 |
| `Nh` | query 头数 | 14 |
| `Nkv` | kv 头数 | 2 |
| `d` | head_dim | 64 |
| `I` | FFN 中间维 | 4864 |
| `V` | 词表大小 | 151936 |

**本模块的所有算子都不带 batch 维。** 每条序列独立计算，批量由运行时层组装。
这和 `KvCache` 的设计一致，见 `forward-pass.md` 第 5 节。

### 2.2 dtype 与精度策略

这是最容易让「和 HF 对不上」的地方，单独定死。

**总原则：镜像 HuggingFace 的每一次 upcast 和 downcast，不自作主张。**
HF 在 bf16/f16 下并不是「全程低精度」，关键位置会临时升到 fp32 再降回去。
我们如果全程用模型 dtype 算，数值会系统性偏离；如果全程 fp32，又和 HF 对不上。
所以必须逐算子照抄它的转换点。

| 算子 | HF 的做法 | 我们要做的 |
| --- | --- | --- |
| `rms_norm` | 升 fp32 算方差与归一化，**降回原 dtype 后**再乘 weight | 同样 |
| `apply_rotary` | cos/sin 在 fp32 生成，**降回原 dtype** 后再参与乘加 | 同样 |
| `silu_and_mul` | 全程原 dtype，不升精度 | 同样 |
| `softmax_last_dim` | 升 fp32 算 softmax，**降回原 dtype** | 同样 |
| `repeat_kv` | 纯搬运，无精度影响 | 同样 |

Phase 1 全程 f32，上表大部分是恒等变换。但**现在就把转换点写进代码**，
否则 Phase 5 切 bf16 时会同时面对「算子不对」和「精度不对」两个问题。

### 2.3 错误处理与命名

- 统一返回 `llm_types::Result<T>`。
- 每个算子先做形状校验，失败返回 `Error::InvalidRequest`，消息里带上实际形状。
  candle 自己的报错信息在形状不匹配时往往看不出是哪一步，不值得依赖。
- 命名与 HF 保持一致（`rms_norm` 而非 `normalize`，`silu_and_mul` 而非
  `swiglu`），减少对照代码时的翻译成本。

### 2.4 参考实现策略

每个算子配两套参照物，在测试里同时使用：

```
   我们的实现 (candle, f32)
        │
        ├── 对比 A：纯 Rust 朴素实现，f64 逐元素循环     <- 真值
        │           放在 ops.rs 的 #[cfg(test)] 里
        │           容差 rtol 1e-5 / atol 1e-6
        │
        └── 对比 B：candle-nn 的等价实现                <- 交叉验证
                    容差 rtol 1e-6 / atol 1e-7
```

对比 A 是**独立于 candle 的真值**：用 f64 逐元素循环算，绝不调用任何被测试的
张量操作。如果 A 和 B 同时不一致，说明我们对 HF 语义的理解有偏差；
如果只有 A 不一致，说明数值精度有问题。这个区分在调试时非常值钱。

`repeat_kv` 没有数值计算，只验证形状与元素对应关系。

---

## 3. 算子规格与执行流程

### 3.1 rms_norm

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>
```

**数学定义**

```
   y = x / sqrt(mean(x^2) + eps) * weight
```

**形状**：`x: (S, H)`，`weight: (H,)`，输出 `(S, H)`。

**执行流程**

```
   ① 记录原件 dtype，把 x 升到 F32
        x32 = x.to_dtype(F32)?

   ② 沿最后一维求平方均值，保持维度
        var = x32.sqr()?.mean_keepdim(D::Minus1)?          (S, 1)

   ③ 取倒数平方根
        denom = (var + eps)?.sqrt()?.recip()?              (S, 1)
        ↑ HF 用的是 rsqrt(var + eps)，不是 1/sqrt(...) 之外的写法

   ④ 归一化，然后降回原 dtype
        normed = x32.broadcast_mul(&denom)?.to_dtype(orig_dtype)?

   ⑤ 乘 weight（在原 dtype 下）
        normed.broadcast_mul(weight)?
```

**精度要点**

第 ④ 步的降回原 dtype **必须在乘 weight 之前**。HF 的写法是
`self.weight * hidden_states.to(input_dtype)`，`weight` 本身就是模型 dtype。
顺序写反在 f32 下看不出来，在 bf16 下会差出可见的误差。

另外注意 `eps` 的类型：`ModelConfig` 里是 `f64`，我们的签名收 `f64`，
但 `candle_nn::ops::rms_norm` 收的是 `f32`。对拍时要做一次 `as f32` 转换，
这是预期内的差异，不是 bug。

**对齐目标**：`transformers` 的 `LlamaRMSNorm.forward` / `Qwen2RMSNorm.forward`。

### 3.2 rotary：拆成两步

RoPE 拆成「建表」与「应用」，因为**表只依赖位置，与张量无关**，
可以算一次反复用。decode 每一步都重建表是纯粹浪费。

```rust
/// 预计算的旋转表，按需增量扩展。
pub struct RotaryTable { cos: Tensor, sin: Tensor }

impl RotaryTable {
    pub fn new(head_dim: usize, theta: f64, dtype: DType, device: &Device) -> Result<Self>;

    /// 取 [0, n) 位置的 cos/sin，形状 (n, d)。
    pub fn slice(&self, n: usize) -> Result<(Tensor, Tensor)>;
}

/// 应用旋转。
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor>
```

**数学定义**（Llama / Qwen 的 non-interleaved 版本）

```
   inv_freq[i] = 1 / theta^(2i/d)          i in 0..d/2
   freqs       = positions 外积 inv_freq   (S, d/2)
   emb         = cat(freqs, freqs)          (S, d)
   cos, sin    = emb.cos(), emb.sin()

   rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
   out = x * cos + rotate_half(x) * sin
```

**建表流程**

```
   ① inv_freq，务必在 F32 下算
        i = [0, 2, 4, ..., d-2]  (d/2 个)
        inv_freq = 1 / theta ^ (i / d)                     (d/2,)

   ② 位置外积
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)   (S, d/2)

   ③ 拼成整头维度
        emb = cat([freqs, freqs], dim = -1)                 (S, d)

   ④ 三角函数
        cos = emb.cos(),  sin = emb.sin()                   (S, d)
```

**应用流程**

```
   ① 拆两半
        x1 = x.narrow(-1, 0, d/2)
        x2 = x.narrow(-1, d/2, d/2)

   ② 构造 rotate_half
        rot = cat([x2.neg()?, x1], dim = -1)                (Nh, S, d)

   ③ 旋转
        out = (x * cos)? + (rot * sin)?                     (Nh, S, d)
        ↑ cos/sin 是 (S, d)，靠广播对齐到 (Nh, S, d)
```

**每次进表前把 cos/sin 降回 x 的 dtype**，这是 HF 的做法。

**精度要点**

`theta` 很大时（Qwen2.5 是 `1e6`），`inv_freq` 的跨度能达到 6 个数量级。
第 ① 步必须在 F32 下做，且**不能用 f64 算完再转**——HF 就是 F32，
我们要对齐的是它，不是数学真值。这一条如果搞反，长序列位置角度的
末位差异会累积成可见的输出偏差。

表本身可以预分配到 `max_position_embeddings`（Qwen2.5 是 32768），
`slice(n)` 只做一次 `narrow`，不重算。

**对拍时注意**：candle-nn 提供两个孪生函数，`rotary_emb::rope` 与
`rotary_emb::rope_i`。前者对应 Llama/Qwen 的 `rotate_half`（前后两半各转一次），
后者对应交错式（相邻两维一组）。**我们要的是 `rope`。**
这两个用错了形状完全一样、结果全错，而且短序列上输出看着还挺合理——
属于那种只在长序列或细致对比时才暴露的 bug。测试里两个都跑一遍，
断言我们与 `rope` 一致、与 `rope_i` 不一致，把这条钉死。

**对齐目标**：`transformers` 的 `Qwen2RotaryEmbedding.forward` +
`apply_rotary_pos_emb`。

### 3.3 silu_and_mul

```rust
pub fn silu_and_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor>
```

**数学定义**

```
   out = silu(gate) * up
   silu(x) = x * sigmoid(x)
```

**形状**：`gate` 与 `up` 都是 `(S, I)`，输出 `(S, I)`。两者形状必须一致。

**执行流程**

```
   ① 校验 gate.shape() == up.shape()
   ② silu(gate)  —— 直接用 Tensor::silu()，这是 candle 的一等 unary op
   ③ 逐元素相乘
```

第 ② 步有一个容易踩的坑：**candle-core 有 `Tensor::silu()`，但没有
`Tensor::sigmoid()`**（`candle-core 0.11`，`src/tensor.rs` 里 `unary_op!` 的
列表中没有 sigmoid）。照 PyTorch 的写法敲 `gate.sigmoid()` 会直接编译不过。

也不建议自己展开成 `1 / (1 + exp(-x))` 再乘：`Tensor::silu()` 会落到后端
的原生实现，比用基础算子拼出来的版本更快，也更贴合后端自己的精度处理。
只有第 5 节那个纯 Rust 参考实现里才需要手写 sigmoid。

**精度要点**：全程原 dtype，不做任何升精度。HF 在 bf16 下就是 bf16 算的。
这是唯一一个不需要转换点的算子。

**对齐目标**：`transformers.activations.SiLU` 与 `Qwen2MLP.forward`。

### 3.4 softmax_last_dim

```rust
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor>
```

**形状**：任意 `(..., N)`，输出同形。实际调用是 `(Nh, S, T)`。

**执行流程**

```
   ① 记录原 dtype，升到 F32
   ② 沿最后一维取最大值，保持维度
        m = x32.max_keepdim(D::Minus1)?                     (..., 1)
   ③ 减最大值后取指数
        e = (x32 - m)?.exp()?
        ↑ 减最大值是必需的：q·kT 除以 sqrt(64) 后仍可能到 ±20，
          不做这个平移会溢出
   ④ 归一化
        y = e.broadcast_div(&e.sum_keepdim(D::Minus1)?)?
   ⑤ 降回原 dtype
```

**精度要点**：第 ① 步的升精度是 HF 的做法
（`softmax(attn_weights, dim=-1, dtype=torch.float32)`），不是我们的优化。
bf16 下不升精度会在长上下文里明显偏离。

**对齐目标**：`transformers` 里 attention 的 softmax 调用。

### 3.5 repeat_kv

```rust
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor>
```

**用途**：GQA。`Nkv=2`、`Nh=14` 时 `n_rep = 7`，每个 KV 头被复制 7 次。

**形状**：`(Nkv, T, d)` 到 `(Nkv * n_rep, T, d)`。

**关键语义：是 repeat_interleave 而不是 tile。**

```
   输入 2 个头，n_rep = 7：

     head 0  ->  heads 0..6      (7 份)
     head 1  ->  heads 7..13     (7 份)

   而不是把整个张量复制 7 遍。
```

写反了形状完全一样，输出全错，而且因为 attention 是对称的，
用随机权重做测试都不一定测得出来——**必须用可区分的输入做测试**
（比如 head 0 全 1、head 1 全 2），见第 5 节。

**执行流程**

```
   ① n_rep == 1 时直接返回，不做任何拷贝
   ② x.unsqueeze(1)                    (Nkv, 1, T, d)
   ③ .expand((Nkv, n_rep, T, d))       ← 视图，不占内存
   ④ .reshape((Nkv * n_rep, T, d))     ← 这里才可能触发拷贝
```

第 ②③④ 步是标准的 broadcast-to-repeat 写法：`expand` 只是把 stride 置 0，
真正的数据复制发生在 `reshape`。Phase 1 不必优化。

**对齐目标**：`transformers.models.llama.modeling_llama.repeat_kv`
（Qwen2 复用了同一份实现）。注意 HF 版本带 batch 维，我们不带。

---

## 4. 调用点

这些算子在模型里的落点，方便对照 `forward-pass.md` 第 4 节：

```
   Attention::forward
        │
        ├── rms_norm          由 DecoderLayer::input_layernorm 调
        ├── apply_rotary      作用于 q 与 k，用在 q_proj/k_proj 之后
        ├── repeat_kv         仅当 Nkv < Nh 时调
        └── softmax_last_dim  作用于 attention scores

   Mlp::forward
        │
        └── silu_and_mul      gate_proj 与 up_proj 之后

   DecoderLayer::forward
        │
        └── rms_norm          两次：input_layernorm 与 post_attention_layernorm

   Model::forward
        │
        └── rms_norm          一次：final norm
```

---

## 5. 验证流程

### 5.1 通用对拍工具

测试模块里放一个共用断言，避免每个算子里各写一遍：

```rust
/// 断言两个张量在容差内一致。
/// 判据：max(|a - b| / (atol + rtol * |b|)) < 1
fn assert_close(actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) -> Result<()>;
```

用相对 + 绝对混合判据而不是纯 `allclose` 的等值判断，这样失败时能打印出
「最差的那个元素在哪、差多少」，比「有元素不一致」有用得多。

### 5.2 每个算子的用例矩阵

| 算子 | 必测用例 |
| --- | --- |
| `rms_norm` | 全零输入（验证 eps 生效）；单元素；大数值输入（验证不溢出） |
| `apply_rotary` | position 0（输出应等于输入）；连续位置；表长度不足时增量扩展 |
| `silu_and_mul` | 大于 88 的输入（sigmoid 饱和）；负值；形状不匹配要报错 |
| `softmax_last_dim` | 含 `+20` 的输入（验证 max 平移）；单元素；和恒为 1 |
| `repeat_kv` | **可区分的输入**（head 各填不同常数）验证是 interleave 而非 tile；`n_rep = 1` 原样返回 |

### 5.3 端到端回归

五个算子单独通过后，跑一次 `models::qwen2` 的 `forward_hidden`，
与 HF 的 `output_hidden_states` 对齐。这是算子集的联合验收：
单算子都对但拼起来错，通常是形状广播或 dtype 转换点的顺序问题。

---

## 6. 执行顺序

自底向上，每步都能独立验证后才进入下一步。

| 步 | 交付物 | 完成判据 |
| --- | --- | --- |
| 1 | `mod.rs` 的 re-export 与 `ops.rs` 骨架 | `cargo test -p llm-core` 通过，模块可编译 |
| 2 | `assert_close` 与随机输入工具 | 故意错一个元素的测试能失败并打印有用信息 |
| 3 | `rms_norm` | 对拍纯 Rust f64 参考与 candle-nn，两者都在容差内 |
| 4 | `softmax_last_dim` | 同上 |
| 5 | `silu_and_mul` | 同上 |
| 6 | `repeat_kv` | 形状 + 元素对应关系测试通过 |
| 7 | `RotaryTable` + `apply_rotary` | position 0 恒等；与参考实现对拍 |
| 8 | 联合回归 | `forward_hidden` 与 HF `hidden_states` 对齐 |

第 3-5 步可以并行做，彼此独立；第 7 步依赖第 6 步的形状约定。

---

## 7. 待确认

**7.1 随机输入的可复现性。**
建议固定种子并用手写的 LCG，不引 `rand`。理由是这个 crate 会被所有 crate
继承依赖，测试用不上一个完整随机数库。这一条是否同意？

**7.2 `RotaryTable` 放哪里。**
本文暂定放在 `tensor::ops`，因为它不含模型知识，只有 `head_dim` / `theta` /
`dtype` 三个参数。另一种选择是放 `models::layers`，因为它更像是「层的状态」
而非「算子」。我倾向前者：`models::layers::Attention` 持有它，但定义在算子层。

**7.3 是否现在就上 `candle` 的 `IndexOp`。**
`mod.rs` 已经 re-export 了 `IndexOp`（切片语法糖）。它让 `narrow` 那类代码
好读很多，但也是唯一会让人写出「隐式拷贝」的接口。建议保留，但在
`Attention` 的热路径上手动用 `narrow` / `reshape`。
