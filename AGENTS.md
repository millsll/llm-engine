# AGENTS.md — llm_engine

Rust 实现的 LLM 推理引擎，架构对标 SGLang（RadixAttention + 连续批处理 + 约束解码）。
本文件是**唯一权威的规划文档**：改架构、改 crate 边界、推进阶段前，先改这里。

## 1. 已定的技术决策

1. **张量层用 candle**（`candle-core` / `candle-nn`）。不自造 Tensor 抽象层，
   也不引入 `cudarc` 之类重写 kernel 层。
2. **`llm-types` 放数据类型与 trait 定义**，不含任何计算逻辑。
3. **`llm-core` 负责模型结构与前向计算**，不认识 KV 存储。
4. **`mem-cache` 负责 cache 实现**，不认识模型。
5. **`llm-runtime` 依赖 core 和 cache**，组织调度器，是唯一负责把两者接起来的地方。

实现顺序：**先把完整前向传播跑通并数值对齐，再补 cache，最后补前缀复用。**

## 2. 目标与非目标

### 目标

- 高吞吐、低延迟的 LLM 服务引擎：连续批处理、PagedAttention、前缀复用。
- OpenAI 兼容 API，流式输出。
- 约束解码（正则 / JSON Schema）与 jump-forward 加速。
- 数值正确性可验证：与 HuggingFace 参考实现逐 token 对齐。

### 非目标（前 5 个阶段不做）

- 训练 / 微调 / 反向传播。
- 非 Transformer 架构。
- 多机多卡；单机多卡放到最后。
- 自己写 CUDA kernel：优先用 candle 及其可选后端，性能不够再考虑。

## 3. 架构对照

| SGLang 组件 | 本项目 | 说明 |
| --- | --- | --- |
| RadixAttention 的 KV 复用 | `mem-cache::radix` | 前缀树 + 引用计数 + LRU 淘汰 |
| MemoryPool | `mem-cache::pool` | 预分配块池，运行期零分配 |
| PagedAttention 存储侧 | `mem-cache::paged` | 实现 `llm-types` 的 `KvCache` trait |
| TokenizerManager / Detokenizer | `llm-tokenizer` | 含增量 UTF-8 解码 |
| ModelRunner | `llm-core` | 模型定义、权重加载、前向计算 |
| Scheduler | `llm-runtime::scheduler` | 组批、抢占、chunked prefill |
| SRT + OpenAI Server | `llm-server` | 唯一的二进制产物 |

## 4. Crate 布局与依赖方向

```
                         llm-types
                 (配置 / 参数 / 错误 / KvCache trait)
                    ↑          ↑          ↑
                    │          │          │
              llm-core    mem-cache   llm-tokenizer
           (模型/前向)   (cache 实现)   (分词)
                    ↑          ↑
                    └────┬─────┘
                         │
                    llm-runtime          ← 唯一同时认识 core 和 cache 的地方
                    (调度/编排)
                         ↑
                    llm-server  (bin)
```

依赖必须单向、无环。允许的组合仅限上图，未画出的组合一律禁止。

| Crate | 职责 | 允许依赖 |
| --- | --- | --- |
| `llm-types` | 配置、采样参数、错误、`BlockId`/`BlockTable`、`KvCache` trait | candle-core、serde |
| `llm-core` | 模型定义、权重加载、前向计算、采样实现、单请求生成循环 | `llm-types`、candle-core、candle-nn、safetensors、hf-hub |
| `mem-cache` | 朴素 cache、块池、分页 cache、前缀树 | `llm-types`、candle-core |
| `llm-tokenizer` | 分词与增量反分词 | `llm-types`、tokenizers |
| `llm-runtime` | 调度器、请求生命周期、驱动前向与 cache | `llm-types`、`llm-core`、`mem-cache`、`llm-tokenizer` |
| `llm-server` | CLI + OpenAI 兼容服务 | 以上全部、axum、tokio、clap |

### 为什么契约放在 `llm-types`

`KvCache` 放在这里，`llm-core` 和 `mem-cache` 才能都看到它却互不依赖。
如果放在 `llm-core`，`mem-cache` 就必须依赖 `llm-core`；如果放在 `mem-cache`，
方向反过来。两种写法都不成环，但都会让一个 crate 被另一个的依赖体积拖累。

这不是理论洁癖，有实测差距（`cargo tree -e normal` 去重后的 crate 数）：

| crate | 依赖树规模 |
| --- | --- |
| `llm-types` | 117 |
| `mem-cache` | **122** |
| `llm-tokenizer` | 127 |
| `llm-core` | 276 |
| `llm-runtime` | 284 |

`llm-core` 之所以重，是因为 `hf-hub` 会拖进 `xet-*` → `reqwest` → `rustls`
→ `aws-lc-rs` 一整条下载栈。存储层不该为这个买单——把契约抽出来后
`mem-cache` 的纯逻辑单测（块池、前缀树、引用计数）编译成本只有原来的 44%。

### 编排权在 `llm-runtime`

`llm-core` 不认识 cache，`mem-cache` 不认识模型，两边都只认识 `llm-types` 的契约。
「这条序列用分页 cache、块表长这样、下一步算哪些 token」这些决策只有运行时层知道。

这也让前向签名保持纯粹：只接收 `&mut dyn KvCache` 和**数据**（token、位置、
块表），不接收**策略**。Phase 5 之后要让注意力算子绕过 gather 直接按块读，
加的也是数据参数，不是把 `mem-cache` 的类型引进 `llm-core`。

### 实际目录

```
crates/
├── llm-types/src/
│   ├── lib.rs
│   ├── error.rs          Error / Result
│   ├── config.rs         ModelConfig（直接反序列化 HF config.json）
│   ├── device.rs         DeviceSpec / DTypeSpec
│   ├── block.rs          BlockId / BlockTable
│   ├── kv.rs             KvCache trait
│   └── sampling.rs       SamplingParams（纯参数）
├── llm-core/src/
│   ├── lib.rs
│   ├── tensor/{mod,ops}.rs       candle 封装 + 自研算子
│   ├── models/{mod,layers,qwen2,llama}.rs
│   ├── weights/{mod,loader,mapping}.rs
│   ├── sampling/mod.rs           采样实现
│   ├── engine.rs                 单请求 prefill/decode 循环
│   └── docs/                     Phase 1 设计基线
│        ├── forward-pass.md      前向传播组件关系
│        └── tensor-ops.md        基础算子执行流程
├── mem-cache/src/
│   ├── lib.rs
│   ├── naive.rs          朴素连续 cache（Phase 1）
│   ├── pool.rs           显存块池（Phase 2）
│   ├── paged.rs          分页 cache（Phase 2）
│   ├── radix.rs          前缀树（Phase 4）
│   └── manager.rs        对外门面：allocate / free / evict / stats
├── llm-tokenizer/src/lib.rs
├── llm-runtime/src/{lib,scheduler}.rs
└── llm-server/src/main.rs
```

## 5. 硬性约束

1. 依赖方向单向、无环。跨 crate 边界前先更新第 4 节的图。
2. `cargo build --workspace` 与 `cargo test --workspace` 必须在**无 GPU 环境**下成功。
   GPU 相关代码只能在 `cuda` feature 下编译，默认 feature 集为 CPU-only。
3. `llm-types` 不得依赖 `llm-core` / `mem-cache` / `llm-runtime`；
   `llm-core` 和 `mem-cache` 不得互相依赖。
4. `unsafe` 在三者中一律禁止（`llm-types` / `llm-core` / `mem-cache`）。
5. 任何需要 GPU 才能执行的新功能，必须有对应的 CPU 参考实现与数值对齐测试。
6. 公共 API 必须有文档注释；库代码禁止 `unwrap()` / `expect()`（测试除外）。

## 6. 构建与测试

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p mem-cache -p llm-types                       # 纯逻辑，秒级
cargo test -p llm-core --features cuda -- --ignored         # 需要 GPU 的用例
cargo build -p llm-server --release
```

**本机现状**：装有 CUDA 13.0 工具链，但 `nvidia-smi` 报错（无驱动）。
CUDA 代码只能编译、无法在本机运行。Phase 1-4 全部要求 CPU 可跑，
到 Phase 5 才需要切到有 GPU 的机器。

**网络**：crates.io 默认被沙箱拦截。`cargo build` / `cargo test` 需要申请联网，
首次拉完依赖后可用 `--offline`。

## 7. 路线图

每个阶段给出目标、交付物、验收标准（DoD）。未通过 DoD 不进入下一阶段。

### Phase 0 — 脚手架（已完成）

virtual workspace 与 6 个 crate 骨架、`AGENTS.md`。
DoD：`cargo build --workspace`、`cargo clippy -D warnings`、`cargo test --workspace` 通过。

### Phase 1 — 完整前向传播（当前阶段）

目标：单请求、CPU、文本进文本出，数值与 HF 对齐。

交付物：

1. `llm-core::weights`：从本地目录加载 Qwen2.5-0.5B 权重。
2. `llm-core::models::layers`：RMSNorm、RoPE、GQA Attention、SwiGLU MLP。
3. `llm-core::models::qwen2`：`forward(tokens, positions, cache) -> logits`。
4. `mem-cache::naive`：朴素连续 cache，实现 `llm_types::KvCache`。
5. `llm-core::sampling`：贪心。
6. `llm-core::engine`：prefill + decode 循环。
7. `llm-tokenizer`：`Tokenizer` 加载与编解码。
8. `llm-server`：命令行 demo。

注意：Phase 1 虽然是朴素 cache，但**前向必须走 `KvCache` trait**，不许把 KV
直接内联在模型里，否则 Phase 2 的替换就不是零成本。

DoD：固定 prompt 下前 20 个生成 token 与 HF transformers 贪心解码完全一致；
f32 下 logits 相对误差 < 1e-3。

### Phase 2 — mem-cache：块池 + 分页 cache

目标：把 KV 从连续张量换成固定块池，语义不变。
交付物：`BlockPool`（预分配、空闲链表、引用计数）、`PagedKvCache`、`CacheManager`。
DoD：`PagedKvCache` 与 `NaiveKvCache` 在同一模型上产出**逐 token 相同**的输出；
块分配与释放无泄漏（引用计数归零后块数回到初始值）。

### Phase 3 — llm-runtime：连续批处理调度

目标：多请求并发。交付物：`llm-runtime::scheduler` 的准入、动态组批、批量前向驱动；
`llm-core::engine` 退化为批量执行器。
DoD：4 并发请求吞吐 > 串行 2x；mock 执行器下调度决策可确定性复现。

### Phase 4 — 前缀树 / RadixAttention

目标：共享前缀零重复计算。交付物：`mem-cache::radix` 前缀树、引用计数、LRU
淘汰、cache-aware 调度（最长前缀优先）。
DoD：共享 system prompt 的多轮对话场景 TTFT 下降 > 50%。

### Phase 5 — GPU 加速

目标：真实硬件性能。交付物：`cuda` feature 打通、bf16、flash-attn 后端接入、
CUDA graph 捕获。
DoD：8B 模型单卡 decode 吞吐达标（基准写入 `benches/baseline.md`）。
前置条件：可用的 GPU 机器。

### Phase 6 — 采样与调度完善

交付物：top-k / top-p / min-p、重复与频率惩罚、logit bias、stop strings、种子；
chunked prefill、抢占策略、请求优先级、取消。
DoD：采样分布与 HF 对齐；长 prompt（> 8k）不 OOM。

### Phase 7 — Server

交付物：axum、`/v1/chat/completions`、`/v1/completions`、SSE 流式、并发限流、
`/metrics`、优雅关闭；`IncrementalDecoder`。
DoD：OpenAI Python client 直连通过；流式输出无 UTF-8 乱码；客户端取消能真正
释放资源。

### Phase 8 — 约束解码

新增 crate `llm-guided`：正则与 JSON Schema 转 FSM、jump-forward、token mask 缓存。
DoD：强制 JSON 输出 100% 可解析；结构化任务 decode 步数减少 > 20%。

### Phase 9 — 分布式与高级优化

张量并行、MoE 专家并行、DP attention、overlap scheduling、投机解码、PD 分离。
DoD：TP=2 相对 TP=1 吞吐提升 > 1.6x。

## 8. 编码规范

- Edition 2024，编译器版本固定在 `rust-toolchain.toml`。
- 错误处理：统一用 `llm_types::Error`；二进制层用 `anyhow`。库中不得吞掉错误。
- 日志用 `tracing`，结构化字段；不打印完整 prompt 内容（除非显式 debug）。
- 命名：类型 `UpperCamelCase`，函数与模块 `snake_case`；模块按领域切分而非按类型
  切分。
- 引入新依赖前先确认编译时间、许可证、是否已有等价实现。**特别地：往
  `llm-types` 加依赖前要想清楚，它会被所有 crate 继承。**
- 提交信息用祈使句，说明「为什么」而不只是「改了什么」。

## 9. 测试策略

- **单元测试**：`llm-types`、`mem-cache`、`llm-runtime::scheduler` 必须可离线跑，
  用确定性种子与 mock 执行器。
- **等价性测试**：`mem-cache` 内部对比 `naive` 与 `paged` 的逐 token 输出。
- **金标准测试**：`tests/golden/` 存放 HF 参考输出的 token id 与 logits。
- **数值对齐**：每个自研算子配 CPU 朴素参考实现，测 rtol / atol。
- **基准**：`criterion`；性能回归会阻止合并。
- **集成测试**：起 `llm-server`，用 HTTP 客户端跑端到端生成。

## 10. Agent 工作规则

1. 动手前先读本文件；改动跨 crate 边界或推进阶段时，先更新本文件再改代码。
2. 每次提交前跑：`cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`。
3. 涉及 `mem-cache` 的改动必须附带块不泄漏的断言；涉及自研算子的改动必须附带
   数值对齐测试。
4. 不要为了让编译通过而放宽第 5 节的硬性约束。
5. 不确定实现路径时，选更小的可验证步骤，并把假设写进 PR 描述。
