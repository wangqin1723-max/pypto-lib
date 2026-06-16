#前置环境（每条命令都需要）

conda activate wq3
export PATH=/usr/local/bin/ptoas-bin:$PATH
export PTOAS_ROOT=/usr/local/bin/ptoas-bin
export PTO_ISA_ROOT=/data/<user>/newpto/pto-isa

export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072 
export PTO2_RING_HEAP=4294967296
unset PTO2_RING_TASK_WINDOW
unset PTO2_RING_DEP_POOL
unset PTO2_RING_HEAP

#4294967296

kill -9 %1

PYPTO_FINITE_PROBE=1 \
task-submit --device 2 --run "python llm/examples/qwen3_14b_npu_generate.py \
    --model-dir /data/<user>/models/Qwen3-14B-4L \
    --prompt '你好' --platform a2a3 \
    --max-seq-len 128 --max-new-tokens 16 \
    > run_4L_perlayer.log 2>&1"


# 5.6号
task-submit --device 6 --run "python -m llm.testing.hf_compare run qwen3_14b.decode \
    -k hf_model_path=/data/<user>/models/Qwen3-14B \
    -k seq_len=128 \
    -k batch=16 \
    -k platform=a2a3 \
    -k hf_dtype=bf16 > run.log 2>&1"


task-submit --device 6 --run "python -m llm.testing.hf_compare run qwen3_14b.decode \
    -k hf_model_path=/data/<user>/models/Qwen3-14B \
    -k seq_len=128 -k batch=16 -k max_seq=512 -k platform=a2a3 2>&1 | tee run.log"

task-submit --device 2 --run "python models/qwen3/14b/qwen3_14b_decode_full.py"


task-submit --device 6 --run "python -m llm.testing.hf_compare run qwen3_14b.decode_full \
    -k hf_model_path=/data/<user>/models/Qwen3-14B \
    -k seq_len=128 -k batch=16 -k max_seq=512 -k platform=a2a3 \
    -k num_layers=8 2>&1 | tee run.log"

# ok
task-submit --device 3 --run "python -m llm.testing.hf_compare run qwen3_14b.decode \
    -k hf_model_path=/data/<user>/models/Qwen3-14B \
    -k seq_len=128 -k batch=16 -k max_seq=128 -k platform=a2a3 2>&1 | tee run_decode.log"     

task-submit --device 6 --run "python -m llm.examples.qwen3_14b_npu_generate \
  --model-dir /data/<user>/models/Qwen3-14B-1L \
  --prompt 'Hello' \
  --max-seq-len 512 \
  --max-new-tokens 16 \
  --profile-verbose 2>&1 | tee run_npu_generate_16t_1L_perf.log"


task-submit --device 5 --run "python models/qwen3/14b/qwen3_14b_decode_full.py \
  -p a2a3\
  --num-layers 1 \
  --max-seq 128 \
  2>&1 | tee run_decode_full_1L_seq64.log"


task-submit --device 5 --run "python -m llm.examples.qwen3_14b_npu_generate \
  --model-dir /data/<user>/models/Qwen3-14B \
  --prompt 'Hello' \
  --max-seq-len 512 \
  --max-new-tokens 16 \
  --profile-verbose 2>&1 | tee run_npu_generate_last_range_logoff.log"


#5.13
#base line
task-submit --device auto --max-time 0 --run "python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--platform a2a3 \
--max-seq-len 640 \
--max-new-tokens 512 \
--l3 &> run_gen_l3_512.log"  


task-submit --device auto --max-time 0 --run "python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--platform a2a3 \
--max-seq-len 640 \
--max-new-tokens 512 \
--l3 \
--profile-verbose 2>&1 | tee run_gen_l3_512_profile.log"  

#进行调试
task-submit --device auto --max-time 0 --run "SIMPLER_CHIP_TIMING=1 \
python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--platform a2a3 \
--max-seq-len 640 \
--max-new-tokens 512 \
--device-id 0 \
> run_gen_l3_512_validate_breakdown.log 2>&1"


#
task-submit --device 6 --max-time 0 --run "python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--max-seq-len 256 \
--max-new-tokens 32 \
--l3 \
--profile-verbose 2>&1 | tee run_swimlane_baseline.log" 

#测试40层kernel的性能
task-submit --device auto --max-time 0 --run "python models/qwen3/14b/qwen3_14b_decode_full.py \
  -p a2a3 \
  --num-layers 4 \
  --max-seq 256 \
  --runtime-profiling"


task-submit --device 6 --max-time 0 --run "python models/qwen3/14b/qwen3_14b_decode.py \
  -p a2a3 \                                  
  --max-seq 4096 \
  --runtime-profiling 2>&1 | tee run_decode_baseline.log"


task-submit --device auto --max-time 0 --run "python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--max-seq-len 640 \
--max-new-tokens 512 \
--profile-verbose 2>&1 | tee run_my_l2.log" 

# 生成32个token是0.75


task-submit --device auto --max-time 0 --run "python llm/examples/qwen3_14b_npu_generate.py \
--model-dir /data/<user>/models/Qwen3-14B \
--prompt 'Huawei is' \
--max-seq-len 640 \
--max-new-tokens 512 \
--profile-verbose 2>&1 | tee run_my_l2.log" 

# 5.18

task-submit --device auto --max-time 1800 --run "python models/deepseek/v4/qkv_proj_rope.py 
-p a2a3 \
-d 0 \
--enable-l2-swimlane \
> 2>&1 | tee qkv_proj_rope_l2_swimlane_520.log " 

#521
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/qkv_proj_rope.py -p a2a3 -d 0 --enable-l2-swimlane > qkv_proj_rope_521_2.log 2>&1" 2>&1 | tail -3


task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_indexer.py -p a2a3 --enable-l2-swimlane > decode_indexer_528_1.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_indexer.py -p a2a3sim"

task-submit --device auto --max-time 1800   --run "python examples/advanced/gemm_eltwise.py -p a2a3 --enable-l2-swimlane > gemm_eltwise_l2_swimlane_522.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_attention_csa.py -p a2a3 --enable-l2-swimlane > decode_attention_csa_l2_swimlane_528_1.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800   --run "python repro_two_accum.py -p a2a3"

# 530
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_qkv_proj_rope.py -p a2a3 --enable-l2-swimlane > qkv_proj_rope_602.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_sparse_attn.py -p a2a3 --enable-l2-swimlane > decode_sparse_attn_609.log 2>&1" 2>&1 | tail -3
#0，0.5
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_attention_csa.py -p a2a3 --enable-l2-swimlane > decode_attention_csa_608.log 2>&1" 2>&1 | tail -3
#1，1
#第二次测又有精度问题？
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_attention_hca.py -p a2a3 --enable-l2-swimlane > decode_attention_hca_601.log 2>&1" 2>&1 | tail -3
#1，0
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_attention_swa.py -p a2a3 --enable-l2-swimlane > decode_attention_swa_601.log 2>&1" 2>&1 | tail -3


#下午
task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_csa.py -p a2a3 --enable-l2-swimlane > decode_csa_601.log 2>&1" 2>&1 | tail -3


task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_indexer_compressor.py -p a2a3 --enable-l2-swimlane > decode_indexer_compressor_601.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/decode_qkv_proj_rope.py -p a2a3 --enable-l2-swimlane > decode_qkv_proj_rope_604.log 2>&1" 2>&1 | tail -3


task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/_tmp_intask_idx_oob.py -p a2a3"

task-submit --device auto --max-time 1800   --run "python models/deepseek/v4/_tmp_rhs_race2.py -p a2a3"

task-submit --device auto --max-time 1800 --run "python models/deepseek/v4/verify_1702_fused_mixx.py -p a2a3 --enable-l2-swimlane > verify_1702_fused_mixx_l2_swimlane_609.log 2>&1" 2>&1 | tail -3


task-submit --device 3 --max-time 1800 --run "python models/deepseek/v4/verify_hc_pre_1spmd_dyn.py -p a2a3 "

# 1. 调详细度:0=DEBUG(最全) 1=INFO 2=WARN 3=ERROR 4=NULL(关)
export ASCEND_GLOBAL_LOG_LEVEL=0

# 2. 二选一,决定日志去哪 ——

#   (a) 直接打屏,跟着终端输出实时看(快速排查最方便)
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#   (b) 或落盘到指定目录(抓 hang / 事后 grep 用)
export ASCEND_PROCESS_LOG_PATH=/tmp/devlog

task-submit --device auto --max-time 1800 --run "python models/deepseek/v4/verify_hc_pre_1spmd.py -p a2a3 --enable-l2-swimlane > verify_hc_pre_1spmd_l2_swimlane_615.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800 --run "python models/deepseek/v4/verify_hc_pre_1spmd.py -p a2a3 --enable-l2-swimlane > verify_hc_pre_1spmd_l2_swimlane_615.log 2>&1" 2>&1 | tail -3

task-submit --device auto --max-time 1800 --run "python models/deepseek/v4/hc_pre.py -p a2a3  --enable-l2-swimlane > hc_pre_fused_draft_l2_swimlane_616.log 2>&1" 2>&1 | tail -3


# 1. 激活新环境 + CANN
conda activate /data/<user>/.conda/envs/pypto20
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=<你的设备号>     # 有卡的设备

# 2. 进算子目录(perf 产物落在“执行时工作目录”下的 output/)
cd /data/<user>/pypto2.0/models/deepseek_v4

# 3. 跑 T=128(对齐 pypto3.0 的 decode 形态),非默认的 16:
python -c "import test_hc_pre as t; t.test_hc_pre(128)"
# 可选 prefill 形态:  t.test_hc_pre(128, True)