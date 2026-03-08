# Qwen3-32B Prefill Kernel Local Tensor Summary

## 1) Plan and Execution

1. Create `examples/qwen3_32b_prefill.py` from decode-style structure, targeting prefill (`batch=16`, `seq<=4096`).
2. Use three major `auto_incore` scopes per token tile:
   - Scope A: RMSNorm + QKV projection
   - Scope B: RoPE + KV cache update + causal attention
   - Scope C: O projection + residual + post RMSNorm + MLP + residual
3. Build with `dump_passes=True` to generate expanded kernels in `qwen3_32b_prefill_dump`.
4. Parse `passes_dump/13_after_AllocateMemoryAddr.py` to estimate local tensor size per InCore function/group.
5. Validate constraints:
   - AIC <= 256KB
   - AIV <= 192KB

## 2) Prefill Program Knobs

- `K_CHUNK=256`
- `Q_OUT_CHUNK=64`
- `KV_OUT_CHUNK=32`
- `SEQ_TILE=120`
- `MLP_OUT_CHUNK=256`
- `TOK_TILE=4`

## 3) Build Result

- Build executed and pass dumps generated successfully up to allocation/report stages.
- Final codegen still hits backend limitation (`No codegen registered for operation: comm.aic_initialize_pipe`), same class of limitation observed in related mixed-kernel experiments.
- Required analysis artifacts are present:
  - `passes_dump/08_after_ExpandMixedKernel.py`
  - `passes_dump/13_after_AllocateMemoryAddr.py`
  - `report/memory_after_AllocateMemoryAddr.txt`

## 4) Local Tensor Statistics (Function-level)

| InCore function | Local tensor total (B) | Buffers |
|---|---:|---:|
| `qwen3_prefill_layer_incore_2_aic` | 248,256 | 17 |
| `qwen3_prefill_layer_incore_0_aic` | 140,288 | 13 |
| `qwen3_prefill_layer_incore_1_aic` | 140,288 | 21 |
| `qwen3_prefill_layer_incore_3_aic` | 140,288 | 13 |
| `qwen3_prefill_layer_incore_3_aiv` | 104,960 | 11 |
| `qwen3_prefill_layer_incore_4_aiv` | 101,376 | 9 |
| `qwen3_prefill_layer_incore_0_aiv` | 72,704 | 13 |
| `qwen3_prefill_layer_incore_2_aiv` | 57,696 | 41 |
| `qwen3_prefill_layer_incore_1_aiv` | 48,128 | 20 |
| `qwen3_prefill_layer_incore_4_aic` | 132,096 | 8 |
| **Total** | **1,186,080** | - |

## 5) Group-level Split (AIC/AIV side-by-side)

| function_group | AIC (B) | AIV (B) | Solo (B) |
|---|---:|---:|---:|
| `qwen3_prefill_layer_incore_2` | 248,256 | 57,696 | 0 |
| `qwen3_prefill_layer_incore_0` | 140,288 | 72,704 | 0 |
| `qwen3_prefill_layer_incore_1` | 140,288 | 48,128 | 0 |
| `qwen3_prefill_layer_incore_3` | 140,288 | 104,960 | 0 |
| `qwen3_prefill_layer_incore_4` | 132,096 | 101,376 | 0 |

## 6) Constraint Check

- **AIC 256KB limit**: PASS (max AIC = `248,256 B`)
- **AIV 192KB limit**: PASS (max AIV = `104,960 B`)

## 7) Tuning Notes

- The current knobs intentionally keep the largest AIC kernel close to, but below, 256KB.
- Structural fusion removed the two tiny solo kernels (`incore_2`, `incore_6` in previous revision).
- Current smallest item is `qwen3_prefill_layer_incore_1_aiv` (`48,128 B`).
- `qwen3_prefill_layer_incore_4_aic` has been uplifted to `132,096 B` (from `17,408 B` in the earlier baseline).
- If further uplift is required, the next target is `incore_4` AIC-side utilization.
