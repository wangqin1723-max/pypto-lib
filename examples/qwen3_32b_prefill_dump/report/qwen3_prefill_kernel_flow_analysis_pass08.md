# Qwen3-32B Prefill Kernel Flow Analysis (Pass 08)

基于 `passes_dump/08_after_ExpandMixedKernel.py`，当前 prefill 已展开为 5 个 mixed kernel group：

- `qwen3_prefill_layer_incore_0_group` (AIC+AIV): Q projection
- `qwen3_prefill_layer_incore_1_group` (AIC+AIV): K/V projection
- `qwen3_prefill_layer_incore_2_group` (AIC+AIV): RoPE + cache update + attention core
- `qwen3_prefill_layer_incore_3_group` (AIC+AIV): O projection + residual
- `qwen3_prefill_layer_incore_4_group` (AIC+AIV): MLP down + final residual/writeback

---

## 1) Top-Level Flow (Orchestration)

```mermaid
flowchart TD
    A[qwen3_prefill_layer]
    A --> B["for b in parallel(0,16)"]
    B --> C["for p0 in range(0,4096,4)"]
    C --> D[Scope-A: sq_sum + inv_rms]
    D --> E["Group0: incore_0_group (Q proj)"]
    E --> F["Group1: incore_1_group (K/V proj)"]
    F --> G["Scope-B: token loop ti=0..3"]
    G --> H["Group2: incore_2_group (RoPE+cache+attn)"]
    H --> I["assemble attn_tile"]
    I --> J["Group3: incore_3_group (O proj + residual)"]
    J --> K["Scope-C: post RMSNorm + gate/up"]
    K --> L["Group4: incore_4_group (down proj + output assemble)"]
    L --> M[out]
```

---

## 2) Group-by-Group Flow Charts

### Group 0: `qwen3_prefill_layer_incore_0_group`

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_0_out / kb
        V0->>C: wq_chunk half + normed half
        V1->>C: wq_chunk half + normed half
        Note over C: assemble full tiles + matmul
        C->>V0: q partial
        C->>V1: q partial
        Note over V0,V1: accumulate q_acc and assemble q_proj_tile
    end
```

### Group 1: `qwen3_prefill_layer_incore_1_group`

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_1_out / kb
        V0->>C: normed half + wk half + wv half
        V1->>C: normed half + wk half + wv half
        Note over C: assemble + matmul(normed,wk/wv)
        C->>V0: k partial + v partial
        C->>V1: k partial + v partial
        Note over V0,V1: accumulate and assemble k_proj_tile/v_proj_tile
    end
```

### Group 2: `qwen3_prefill_layer_incore_2_group`

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop h_0_out (head-block)
        Note over V0,V1: q head -> RoPE
        V0->>C: q_rot / cache tile halves
        V1->>C: q_rot / cache tile halves
        Note over C: assemble K/V tiles, compute scores
        Note over V0,V1: softmax branch + online li/mi/oi update
        C->>V0: intermediate partials
        C->>V1: intermediate partials
        Note over V0,V1: finalize ctx, assemble attn_row
    end
```

### Group 3: `qwen3_prefill_layer_incore_3_group`

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_2_out / kb
        V0->>C: attn chunk half + wo chunk half
        V1->>C: attn chunk half + wo chunk half
        Note over C: assemble + matmul
        C->>V0: o partial
        C->>V1: o partial
        Note over V0,V1: +residual, assemble resid1_tile
    end
```

### Group 4: `qwen3_prefill_layer_incore_4_group`

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop dob_0_out
        V0->>C: w_down chunk half
        V1->>C: w_down chunk half
        Note over C: assemble full w_down chunk + matmul(mlp_chunk)
        C->>V0: down partial
        C->>V1: down partial
        Note over V0,V1: down_prev + partial
        Note over V0,V1: assemble down_proj_tile and out
    end
```

---

## 3) Notes

- 当前 pass 8 结果下，原先特别小的 solo kernel 已被并入 mixed group，不再单独存在。
- 从执行路径看，`incore_2_group` 仍是最复杂链路（RoPE、cache、attention、online 更新交织），是后续性能调优优先点。
- `incore_4_group` 已显著放大（AIC/AIV 均提升），并承担最终输出写回路径。
