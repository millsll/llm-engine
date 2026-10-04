# llm-core 前向传播：组件关系

本文只描述 **`llm-core` 内部**前向传播的组件划分与数据流，供审查。
跨 crate 的边界只在第 2 节标注，其余不涉及。

绘图约定：**任何带右边框的方框，框内只写 ASCII**。终端与 Markdown 渲染器
对中日韩字符的宽度处理不一致，框内混入宽字符会让右边框错位。所有中文说明
一律放在框外。无右边框的示意图（树形）不受此限。

目标模型 Qwen2.5-0.5B，下文出现的具体数字都取自它的配置。

| 参数 | 符号 | 值 |
| --- | --- | --- |
| `hidden_size` | H | 896 |
| `num_hidden_layers` | L | 24 |
| `num_attention_heads` | — | 14 |
| `num_key_value_heads` | — | 2 |
| `head_dim` | d | 64 |
| `intermediate_size` | — | 4864 |
| `vocab_size` | V | 151936 |
| `tie_word_embeddings` | — | true |

---

## 1. 组件清单与职责

| 组件 | 路径 | 职责 | 不做什么 |
| --- | --- | --- | --- |
| `tensor` | `src/tensor/mod.rs` | 统一 re-export candle 的 Tensor / Device / DType | 不做计算 |
| `tensor::ops` | `src/tensor/ops.rs` | 自研算子：RMSNorm / RoPE / SwiGLU / softmax / repeat_kv | 不持有状态 |
| `models::layers` | `src/models/layers.rs` | RmsNorm、RotaryEmbedding、Attention、Mlp、DecoderLayer | 不管权重从哪来 |
| `models::qwen2` | `src/models/qwen2.rs` | 组装整模型：embedding + L×DecoderLayer + final norm + lm_head | 不做采样 |
| `weights::loader` | `src/weights/loader.rs` | safetensors 加载、dtype 转换、组装 VarBuilder | 不认识层结构 |
| `weights::mapping` | `src/weights/mapping.rs` | HF 权重名与内部参数名的映射、未匹配检查 | 不读文件 |
| `sampling` | `src/sampling/mod.rs` | logits 到下一个 token | 不碰 KV cache |
| `engine` | `src/engine.rs` | prefill / decode 循环，串起上面所有组件 | 不做调度（Phase 3 起归 llm-runtime） |

---

## 2. 跨 crate 契约边界

`llm-core` 只通过 `llm-types` 与外界交互，**不认识 `mem-cache`，也不认识 `llm-runtime`**。

```
   ┌─────────────┐
   │  llm-types  │
   └─────────────┘
          │
          ├── ModelConfig / DeviceSpec / DTypeSpec ───► llm-core::models::qwen2
          │                                             llm-core::weights::loader
          │
          ├── SamplingParams ────────────────────────► llm-core::sampling
          │
          └── KvCache (trait) ◄──────────────────────► llm-core::models::layers
                                                              ▲        ::Attention
                        impl by mem-cache::NaiveKvCache        │
                        impl by mem-cache::PagedKvCache ───────┘
                                  ▲
                                  │ passed as &mut dyn KvCache
                                  │ by llm-runtime when it drives forward
                           llm-runtime
```

**关键不变量：`llm-core` 的代码里不出现 `mem_cache::` 或 `llm_runtime::`。**

---

## 3. 静态组件依赖（模块级）

```
   engine
     ├── sampling
     │
     ├── models::qwen2
     │     └── models::layers          <- 唯一接触 dyn KvCache 的模块
     │           └── tensor::ops       <- RMSNorm / RoPE / SwiGLU / softmax / repeat_kv
     │                 └── tensor      <- candle 的 Tensor / Device / DType
     │
     └── weights::loader
           └── weights::mapping
```

依赖方向自上而下，**不允许反向**。三点要特别注意：

- `models::layers` 是**唯一**接触 `dyn KvCache` 的模块。
- `tensor::ops` 在最底层，不认识模型、cache、配置。
- `weights::*` 不依赖 `models::*`。它只产出 `VarBuilder`，由 `models::qwen2`
  在构造时消费——是数据依赖，不是调用依赖，所以不在树里画成孩子的兄弟。

---

## 4. 一次 prefill 的数据流

输入 `tokens: [S]` 与 `positions: [S]`，输出 `logits: [S, V]`。

```
   tokens [S]
      │
      ▼  embed_tokens.weight [V, H]
   hidden [S, H]
      │
      │  ┌─ DecoderLayer × L ────────────────────────────────────
      │  │
      │  │   input_layernorm  (RMSNorm)
      │  │        │
      │  │        ▼
      │  │   q_proj / k_proj / v_proj
      │  │        │
      │  │        ▼
      │  │   RoPE(q, k, positions)
      │  │        │
      │  │        ├───────────────► cache.append(i, k, v)       [write]
      │  │        │
      │  │        ▼
      │  │   k, v = cache.get(i)     includes all past tokens   [read]
      │  │        │
      │  │        ▼
      │  │   repeat_kv   (GQA: kv_heads 2 ──► 14)
      │  │        │
      │  │        ▼
      │  │   attention: softmax(q·kT / sqrt(64) + causal_mask) · v
      │  │        │
      │  │        ▼
      │  │   o_proj  ──►  (+)  ◄── residual
      │  │        │
      │  │        ▼
      │  │   post_attention_layernorm
      │  │        │
      │  │        ▼
      │  │   MLP: down( silu(gate) · up )  ──►  (+)  ◄── residual
      │  │
      │  └───────────────────────────────────────────────────────
      │
      ▼
   final norm (RMSNorm)
      │
      ▼  lm_head   (tie_word_embeddings=true 时复用 embed_tokens.weight)
   logits [S, V]
```

顺序要点：`cache.get(i)` 返回的是**含本步新写入在内**的全部历史 K/V，
所以 `append` 必须排在 `get` 之前。这个顺序写死在 `Attention::forward` 里。

---

## 5. 张量形状流转

`S` 是本次前向的 token 数，`T` 是 cache 中已有的 token 数，batch 维一律省略
（单序列）。

| 阶段 | prefill（T=0, S=n） | decode（T=t, S=1） |
| --- | --- | --- |
| `tokens` | `[n]` | `[1]` |
| `hidden` | `[n, 896]` | `[1, 896]` |
| `q` reshape + transpose | `[14, n, 64]` | `[14, 1, 64]` |
| `k` / `v` reshape + transpose | `[2, n, 64]` | `[2, 1, 64]` |
| `cache.get(i)` 返回 | `[2, n, 64]` | `[2, t+1, 64]` |
| `repeat_kv` 之后 | `[14, n, 64]` | `[14, t+1, 64]` |
| attention scores | `[14, n, n]` | `[14, 1, t+1]` |
| attention 输出 merge heads | `[n, 896]` | `[1, 896]` |
| SwiGLU 中间（gate / up） | `[n, 4864]` | `[1, 4864]` |
| `logits` | `[n, 151936]` | `[1, 151936]` |

**形状上唯一随 cache 变化的量是 `T`。** prefill 与 decode 走同一份代码，
区别只有 `S` 和 `T`。这是必须保持的性质——不要为 decode 单写一条路径。

---

## 6. KV cache 的读写点

```
   ┌──────────────── DecoderLayer i ──────────────────┐
   │                                                  │
   │   Attention                                      │
   │      ├── cache.append(i, k, v)    (the only      │
   │      │                            write point)   │
   │      └── cache.get(i)             (the only      │
   │                                    read point)   │
   │                                                  │
   │   RmsNorm / Mlp / residual        (never touch)  │
   └──────────────────────────────────────────────────┘
```

完全不碰 cache 的组件：`embed_tokens`、`input_layernorm`、`o_proj`、
`post_attention_layernorm`、MLP 的三个投影、`final norm`、`lm_head`、`sampling`。

这条边界是 Phase 2「朴素 cache 换成分页 cache、模型代码零改动」的前提。

---

## 7. 权重文件到组件的映射

由 `weights::mapping` 负责，`weights::loader` 按此组装。

| HF 权重名 | 落入组件 |
| --- | --- |
| `model.embed_tokens.weight` | `Model::embed_tokens` |
| `model.layers.{i}.input_layernorm.weight` | `DecoderLayer::input_layernorm` |
| `model.layers.{i}.self_attn.q_proj.{weight,bias}` | `Attention::q_proj` |
| `model.layers.{i}.self_attn.k_proj.{weight,bias}` | `Attention::k_proj` |
| `model.layers.{i}.self_attn.v_proj.{weight,bias}` | `Attention::v_proj` |
| `model.layers.{i}.self_attn.o_proj.weight` | `Attention::o_proj` |
| `model.layers.{i}.post_attention_layernorm.weight` | `DecoderLayer::post_attention_layernorm` |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `Mlp::{gate,up,down}_proj` |
| `model.norm.weight` | `Model::norm` |
| `lm_head.weight` | `Model::lm_head`（tied 时文件里不存在，复用 embed） |

Qwen2 的 q/k/v **带 bias**，Llama 不带。这个差异收敛在
`models::layers::Attention` 的构造参数 `qkv_bias: bool` 里，不扩散到别处。

---

## 8. engine 循环

```
   tokens = tokenizer.encode(prompt)
      │
      ▼
   forward(tokens, positions = 0..S, cache)          <- prefill
      │
      ▼
   logits[last] ──► sampling ──► next_token
      │
      ▼
   loop, at most max_tokens times:
      │
      ├─ forward([next_token], positions = S + step, cache)   <- decode
      │     │
      │     ▼
      │  logits[0] ──► sampling ──► token
      │     │
      │     └─ 命中 EOS 或 stop string 则跳出
      │
      └─ 否则把 token 追加进输出，继续
      │
      ▼
   tokenizer.decode(生成的 tokens)
```

Phase 3 起，这个循环的驱动权移交给 `llm-runtime::scheduler`，
`engine` 退化为「给定一批序列执行一次前向」的批量执行器。
前向本身不因这次移交而改变——这正是第 6 节那条边界要守住的东西。

---

## 9. 待确认的设计点

以下四点会决定 Phase 1 的接口形状，请在动手前确认。

### 9.1 positions 是否作为参数传入

建议传入。prefill 用 `0..S`，decode 用 `cache.seq_len()..seq_len()+1`，
看起来可以由 `Attention` 从 cache 推算，但现在省下的这点事，会在
chunked prefill（Phase 6）和投机解码（Phase 9）需要非连续位置时还回来。

### 9.2 append 与 get 由谁调用

建议由 `Attention` 每层自己调用，对应 `KvCache::append(layer, k, v)`。
备选是模型层收集全部层的 K/V 后统一批量写。前者简单，后者对分页 cache
更友好（一次分散写入，而不是逐层触发）。

**这条直接决定 `KvCache` trait 的形状，需要先定。**

### 9.3 forward 返回全序列 logits 还是只算末位

建议拆成两步：

```rust
fn forward_hidden(&self, tokens: &Tensor, positions: &Tensor, cache: &mut dyn KvCache)
    -> Result<Tensor>;                                  // [S, H]
fn logits(&self, hidden: &Tensor) -> Result<Tensor>;    // [S, V]
```

prefill 时只对末位调 `logits`。理由很实际：S=1000 时 `[1000, 151936]` 的
f32 张量是 600MB，而引擎只需要最后一行。拆开之后由调用方决定切哪一段，
同时「与 HF 全量 logits 对齐」的测试也能直接调 `logits` 取全量。

### 9.4 dtype 从哪里来

权重文件里的 dtype 与运行 dtype 可能不同（HF 官方权重是 bf16）。
建议在 `Model::new(cfg, vb, dtype)` 显式传入运行精度，加载时统一转换，
不在层内做隐式转换。

---

## 10. Phase 1 实现顺序

按依赖方向自底向上，每一步都能独立验证。

| 步 | 交付物 | 验证方式 |
| --- | --- | --- |
| 1 | `weights::mapping` + `weights::loader` | 加载 Qwen2.5-0.5B，断言权重名与参数全部命中 |
| 2 | `tensor::ops` 的 rms_norm / rotary / silu_and_mul | 与 candle-nn 等价实现做数值对齐 |
| 3 | `models::layers::Attention`（含 GQA） | 固定随机权重，与手写参考实现逐元素对拍 |
| 4 | `models::qwen2::Model::forward_hidden` | 与 HF 的 `output_hidden_states` 对齐 |
| 5 | `mem-cache::naive::NaiveKvCache` | 与「不用 cache 每步重算全序列」结果一致 |
| 6 | `lm_head` + `sampling` | 与 HF 贪心解码前 20 个 token 逐位一致 |
| 7 | `engine` + `llm-tokenizer` + CLI | 端到端跑通 |
