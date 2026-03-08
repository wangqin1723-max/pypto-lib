# pypto.program: Qwen3SingleLayerPrefill
import pypto.language as pl

@pl.program
class Qwen3SingleLayerPrefill:
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_prefill_layer(self, hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], out_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]) -> pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]:
        for b_0, (k_cache_iter_1, out_iter_1, v_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(k_cache_0, out_0, v_cache_0), chunk=4):
            for p0_0, (k_cache_iter_3, out_iter_3, v_cache_iter_3) in pl.range(0, 4096, 4, init_values=(k_cache_iter_1, out_iter_1, v_cache_iter_1)):
                sq_sum_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
                for kb_0, (sq_sum_iter_2,) in pl.range(0, 20, 1, init_values=(sq_sum_1,)):
                    k0_0: pl.Scalar[pl.INDEX] = kb_0 * 256
                    _t0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 256], [b_0, p0_0, k0_0])
                    x_chunk_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
                    _t1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(x_chunk_0, x_chunk_0)
                    _t2: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(_t1)
                    sq_sum_4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, _t2)
                    sq_sum_3: pl.Tensor[[4, 1], pl.FP32] = pl.yield_(sq_sum_4)
                _t3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_3, 0.000195313)
                _t4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(_t3, 1e-06)
                inv_rms_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(_t4)
                q_proj_tile_0: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.create([4, 5120], dtype=pl.BFLOAT16)
                k_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.create([4, 1024], dtype=pl.BFLOAT16)
                v_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.create([4, 1024], dtype=pl.BFLOAT16)
                for ob_0_out, (k0_iter_1_outer_l0, kb_iter_1_outer_l0, q_proj_tile_iter_1_outer_l0, x_chunk_iter_1_outer_l0) in pl.range(0, 10, 1, init_values=(k0_0, kb_0, q_proj_tile_0, x_chunk_0)):
                    ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]) = self.call_group(qwen3_prefill_layer_incore_0_group, b_0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_0, k0_iter_1_outer_l0, kb_0, kb_iter_1_outer_l0, ob_0_out, p0_0, q_proj_tile_0, q_proj_tile_iter_1_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1_outer_l0)
                    k0_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                    kb_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                    q_proj_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 5120], pl.BFLOAT16] = ret[2]
                    x_chunk_iter_1_outer_l1_rv: pl.Tensor[[4, 256], pl.FP32] = ret[3]
                    k0_iter_1_outer_l0_rv, kb_iter_1_outer_l0_rv, q_proj_tile_iter_1_outer_l0_rv, x_chunk_iter_1_outer_l0_rv = pl.yield_(k0_iter_1_outer_l1_rv, kb_iter_1_outer_l1_rv, q_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_1_outer_l1_rv)
                for ob_1_out, (gamma_iter_1_outer_l0, k0_iter_6_outer_l0, k_proj_tile_iter_1_outer_l0, kb_iter_4_outer_l0, normed_iter_1_outer_l0, v_proj_tile_iter_1_outer_l0, x_chunk_iter_6_outer_l0) in pl.range(0, 4, 1, init_values=(gamma_0, k0_2, k_proj_tile_0, kb_2, normed_0, v_proj_tile_0, x_chunk_2)):
                    ret: pl.Tuple([pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]) = self.call_group(qwen3_prefill_layer_incore_1_group, b_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_2, k0_iter_6_outer_l0, k_proj_tile_0, k_proj_tile_iter_1_outer_l0, kb_2, kb_iter_4_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, p0_0, v_proj_tile_0, v_proj_tile_iter_1_outer_l0, wk_0, wv_0, x_chunk_2, x_chunk_iter_6_outer_l0)
                    gamma_iter_1_outer_l1_rv: pl.Tensor[[1, 256], pl.FP32] = ret[0]
                    k0_iter_6_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                    k_proj_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 1024], pl.BFLOAT16] = ret[2]
                    kb_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[3]
                    normed_iter_1_outer_l1_rv: pl.Tensor[[4, 256], pl.FP32] = ret[4]
                    v_proj_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 1024], pl.BFLOAT16] = ret[5]
                    x_chunk_iter_6_outer_l1_rv: pl.Tensor[[4, 256], pl.FP32] = ret[6]
                    gamma_iter_1_outer_l0_rv, k0_iter_6_outer_l0_rv, k_proj_tile_iter_1_outer_l0_rv, kb_iter_4_outer_l0_rv, normed_iter_1_outer_l0_rv, v_proj_tile_iter_1_outer_l0_rv, x_chunk_iter_6_outer_l0_rv = pl.yield_(gamma_iter_1_outer_l1_rv, k0_iter_6_outer_l1_rv, k_proj_tile_iter_1_outer_l1_rv, kb_iter_4_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_6_outer_l1_rv)
                attn_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                attn_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.mul(attn_tile_0, 0.0)
                for ti_0, (attn_tile_iter_2, k_cache_iter_5, v_cache_iter_5) in pl.range(0, 4, 1, init_values=(attn_tile_1, k_cache_iter_3, v_cache_iter_3)):
                    pos_0: pl.Scalar[pl.INDEX] = p0_0 + ti_0
                    ctx_len_0: pl.Scalar[pl.INDEX] = pos_0 + 1
                    ctx_blocks_0: pl.Scalar[pl.INDEX] = (ctx_len_0 + 120 - 1) // 120
                    cos_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_cos_0, [1, 128], [pos_0, 0])
                    sin_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_sin_0, [1, 128], [pos_0, 0])
                    cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 0])
                    cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 128 // 2])
                    sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 0])
                    sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 128 // 2])
                    attn_row_0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32)
                    attn_row_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.mul(attn_row_0, 0.0)
                    for h_0_out, (attn_row_iter_2_outer_l0, k_cache_iter_7_outer_l0, v_cache_iter_7_outer_l0) in pl.range(0, 8, 1, init_values=(attn_row_1, k_cache_iter_5, v_cache_iter_5)):
                        ret: pl.Tuple([pl.Tensor[[1, 5120], pl.FP32], pl.Tensor[[524288, 128], pl.BFLOAT16], pl.Tensor[[524288, 128], pl.BFLOAT16]]) = self.call_group(qwen3_prefill_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_0, k_cache_iter_1, k_cache_iter_3, k_cache_iter_5, k_cache_iter_7_outer_l0, k_proj_tile_iter_1_outer_l0_rv, pos_0, q_proj_tile_iter_1_outer_l0_rv, sin_hi_0, sin_lo_0, ti_0, v_cache_0, v_cache_iter_1, v_cache_iter_3, v_cache_iter_5, v_cache_iter_7_outer_l0, v_proj_tile_iter_1_outer_l0_rv)
                        attn_row_iter_2_outer_l1_rv: pl.Tensor[[1, 5120], pl.FP32] = ret[0]
                        k_cache_iter_7_outer_l1_rv: pl.Tensor[[524288, 128], pl.BFLOAT16] = ret[1]
                        v_cache_iter_7_outer_l1_rv: pl.Tensor[[524288, 128], pl.BFLOAT16] = ret[2]
                        attn_row_iter_2_outer_l0_rv, k_cache_iter_7_outer_l0_rv, v_cache_iter_7_outer_l0_rv = pl.yield_(attn_row_iter_2_outer_l1_rv, k_cache_iter_7_outer_l1_rv, v_cache_iter_7_outer_l1_rv)
                    attn_tile_4: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(attn_tile_iter_2, attn_row_iter_2_outer_l0_rv, [ti_0, 0])
                    attn_tile_3, k_cache_6, v_cache_6 = pl.yield_(attn_tile_4, k_cache_iter_7_outer_l0_rv, v_cache_iter_7_outer_l0_rv)
                resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                for ob_2_out, (k0_iter_11_outer_l0, kb_iter_7_outer_l0, resid1_tile_iter_1_outer_l0) in pl.range(0, 10, 1, init_values=(k0_7, kb_5, resid1_tile_0)):
                    ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]) = self.call_group(qwen3_prefill_layer_incore_3_group, attn_tile_3, b_0, hidden_states_0, k0_7, k0_iter_11_outer_l0, kb_5, kb_iter_7_outer_l0, ob_2_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
                    k0_iter_11_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                    kb_iter_7_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                    resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 5120], pl.FP32] = ret[2]
                    k0_iter_11_outer_l0_rv, kb_iter_7_outer_l0_rv, resid1_tile_iter_1_outer_l0_rv = pl.yield_(k0_iter_11_outer_l1_rv, kb_iter_7_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv)
                sq_sum_5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum_6: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_5, 0.0)
                for kb_10, (k0_iter_16, sq_sum_iter_7, x_chunk_iter_11) in pl.range(0, 20, 1, init_values=(k0_iter_11_outer_l0_rv, sq_sum_6, x_chunk_iter_6_outer_l0_rv)):
                    k0_18: pl.Scalar[pl.INDEX] = kb_10 * 256
                    x_chunk_13: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 256], [0, k0_18])
                    _t47: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(x_chunk_13, x_chunk_13)
                    _t48: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(_t47)
                    sq_sum_9: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_7, _t48)
                    k0_17, sq_sum_8, x_chunk_12 = pl.yield_(k0_18, sq_sum_9, x_chunk_13)
                _t49: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_8, 0.000195313)
                _t50: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(_t49, 1e-06)
                inv_rms_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(_t50)
                post_norm_tile_0: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.create([4, 5120], dtype=pl.BFLOAT16)
                down_proj_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.mul(down_proj_tile_0, 0.0)
                for kb_11, (gamma_iter_6, k0_iter_19, normed_iter_6, post_norm_tile_iter_1, x_chunk_iter_14) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_l0_rv, k0_17, normed_iter_1_outer_l0_rv, post_norm_tile_0, x_chunk_12)):
                    k0_21: pl.Scalar[pl.INDEX] = kb_11 * 256
                    x_chunk_16: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 256], [0, k0_21])
                    gamma_8: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(post_rms_weight_0, [1, 256], [0, k0_21])
                    _t51: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_16, inv_rms_1)
                    normed_8: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t51, gamma_8)
                    _t52: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(normed_8, target_type=pl.BFLOAT16, mode=2)
                    post_norm_tile_3: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile_iter_1, _t52, [0, k0_21])
                    gamma_7, k0_20, normed_7, post_norm_tile_2, x_chunk_15 = pl.yield_(gamma_8, k0_21, normed_8, post_norm_tile_3, x_chunk_16)
                for ob_3, (down_proj_tile_iter_2, k0_iter_22, kb_iter_12, o0_iter_1, out_iter_5) in pl.range(0, 100, 1, init_values=(down_proj_tile_1, k0_20, kb_11, o0_0, out_iter_3)):
                    o0_3: pl.Scalar[pl.INDEX] = ob_3 * 256
                    gate_acc_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32)
                    up_acc_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32)
                    gate_acc_1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(gate_acc_0, 0.0)
                    up_acc_1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(up_acc_0, 0.0)
                    for kb_14, (gate_acc_iter_2, k0_iter_24, up_acc_iter_2) in pl.range(0, 20, 1, init_values=(gate_acc_1, k0_iter_22, up_acc_1)):
                        k0_26: pl.Scalar[pl.INDEX] = kb_14 * 256
                        post_chunk_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(post_norm_tile_2, [4, 256], [0, k0_26])
                        wg_0: pl.Tensor[[256, 256], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [256, 256], [k0_26, o0_3])
                        wu_0: pl.Tensor[[256, 256], pl.BFLOAT16] = pl.tensor.view(w_up_0, [256, 256], [k0_26, o0_3])
                        _t53: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        gate_acc_4: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.add(gate_acc_iter_2, _t53)
                        _t54: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        up_acc_4: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.add(up_acc_iter_2, _t54)
                        gate_acc_3, k0_25, up_acc_3 = pl.yield_(gate_acc_4, k0_26, up_acc_4)
                    _t55: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.neg(gate_acc_3)
                    _t56: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.exp(_t55)
                    _t57: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.add(_t56, 1.0)
                    sigmoid_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.recip(_t57)
                    _t58: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(gate_acc_3, sigmoid_0)
                    mlp_chunk_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(_t58, up_acc_3)
                    mlp_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
                    for dob_0_out, (down_proj_tile_iter_4_outer_l0, out_iter_7_outer_l0) in pl.range(0, 10, 1, init_values=(down_proj_tile_iter_2, out_iter_5)):
                        ret: pl.Tuple([pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]]) = self.call_group(qwen3_prefill_layer_incore_4_group, b_0, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, ob_3, out_0, out_iter_1, out_iter_3, out_iter_5, out_iter_7_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv, w_down_0)
                        down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 5120], pl.FP32] = ret[0]
                        out_iter_7_outer_l1_rv: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = ret[1]
                        down_proj_tile_iter_4_outer_l0_rv, out_iter_7_outer_l0_rv = pl.yield_(down_proj_tile_iter_4_outer_l1_rv, out_iter_7_outer_l1_rv)
                    down_proj_tile_3, k0_23, kb_13, o0_2, out_6 = pl.yield_(down_proj_tile_iter_4_outer_l0_rv, k0_25, kb_14, o0_3, out_iter_7_outer_l0_rv)
                k_cache_4, out_4, v_cache_4 = pl.yield_(k_cache_6, out_6, v_cache_6)
            k_cache_2, out_2, v_cache_2 = pl.yield_(k_cache_4, out_4, v_cache_4)
        return out_2
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_0_aic(self, b_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_0: pl.Tensor[[4, 1], pl.FP32], k0_0: pl.Scalar[pl.INDEX], k0_iter_1_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], q_proj_tile_0: pl.Tensor[[4, 5120], pl.BFLOAT16], q_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.BFLOAT16], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], x_chunk_0: pl.Tensor[[4, 256], pl.FP32], x_chunk_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_0_in, (k0_iter_1_outer_l1, kb_iter_1_outer_l1, q_proj_tile_iter_1_outer_l1, x_chunk_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_1_outer_l0, kb_iter_1_outer_l0, q_proj_tile_iter_1_outer_l0, x_chunk_iter_1_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
            for kb_3, (k0_iter_3, x_chunk_iter_3) in pl.range(0, 20, 1, init_values=(k0_iter_1_outer_l1, x_chunk_iter_1_outer_l1)):
                k0_5: pl.Scalar[pl.INDEX] = kb_3 * 256
                wq_chunk_0__h0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wq_chunk_0__h1: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wq_chunk_0__tmp: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.create(__list__(256, 64), dtype=pl.BFLOAT16)
                wq_chunk_0__mid: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(wq_chunk_0__tmp, wq_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wq_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(wq_chunk_0__mid, wq_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t7__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                _t7__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                _t7__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                _t7__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(_t7__tmp, _t7__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                _t7: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(_t7__mid, _t7__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                _t8: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(_t7, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t8, __list__(2, 64), __list__(0, 0))
                __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t8, __list__(2, 64), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_4, x_chunk_4 = pl.yield_(k0_5)
            k0_iter_1_outer_l1_rv, kb_iter_1_outer_l1_rv, q_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_1_outer_l1_rv = pl.yield_(k0_4, kb_3, x_chunk_4)
        return k0_iter_1_outer_l1_rv, kb_iter_1_outer_l1_rv, q_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_0_aiv(self, b_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_0: pl.Tensor[[4, 1], pl.FP32], k0_0: pl.Scalar[pl.INDEX], k0_iter_1_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], q_proj_tile_0: pl.Tensor[[4, 5120], pl.BFLOAT16], q_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.BFLOAT16], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], x_chunk_0: pl.Tensor[[4, 256], pl.FP32], x_chunk_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]:
        inv_rms_0_tile: pl.Tile[[4, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[4, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(inv_rms_0, [0, 0], [4, 1], [4, 1], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for ob_0_in, (k0_iter_1_outer_l1, kb_iter_1_outer_l1, q_proj_tile_iter_1_outer_l1, x_chunk_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_1_outer_l0, kb_iter_1_outer_l0, q_proj_tile_iter_1_outer_l0, x_chunk_iter_1_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
            q_acc_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.create([2, 64], dtype=pl.FP32)
            q_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(q_acc_0, 0.0)
            for kb_3, (k0_iter_3, q_acc_iter_2, x_chunk_iter_3) in pl.range(0, 20, 1, init_values=(k0_iter_1_outer_l1, q_acc_1, x_chunk_iter_1_outer_l1)):
                k0_5: pl.Scalar[pl.INDEX] = kb_3 * 256
                _t5: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 128], [b_0, p0_0 + AIV_IDX * 128, k0_5])
                x_chunk_5: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(_t5, target_type=pl.FP32, mode=2)
                gamma_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_5 + AIV_IDX * 128])
                _t6: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_5, inv_rms_0)
                normed_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t6, gamma_0)
                wq_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wq_0, [128, 64], [k0_5 + AIV_IDX * 128, q0_0])
                pl.comm.tpush_to_aic(wq_chunk_0, AIV_IDX)
                _t7: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(_t7, AIV_IDX)
                _t8: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                q_acc_4: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(q_acc_iter_2, _t8)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_4, q_acc_3, x_chunk_4 = pl.yield_(k0_5, q_acc_4, x_chunk_5)
            _t9: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2)
            q_proj_tile_3: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile_iter_1_outer_l1, _t9, [0 + AIV_IDX * 2, q0_0])
            k0_iter_1_outer_l1_rv, kb_iter_1_outer_l1_rv, q_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_1_outer_l1_rv = pl.yield_(k0_4, kb_3, q_proj_tile_3, x_chunk_4)
        return k0_iter_1_outer_l1_rv, kb_iter_1_outer_l1_rv, q_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_prefill_layer_incore_0_aic", aiv="qwen3_prefill_layer_incore_0_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_prefill_layer_incore_0_group:
        """Parameter passing:
          call_group(qwen3_prefill_layer_incore_0_group, b_0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_0, k0_iter_1_outer_l0, kb_0, kb_iter_1_outer_l0, ob_0_out, p0_0, q_proj_tile_0, q_proj_tile_iter_1_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1_outer_l0)
            → qwen3_prefill_layer_incore_0_aic(b_0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_0, k0_iter_1_outer_l0, kb_0, kb_iter_1_outer_l0, ob_0_out, p0_0, q_proj_tile_0, q_proj_tile_iter_1_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1_outer_l0)
            → qwen3_prefill_layer_incore_0_aiv(b_0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_0, k0_iter_1_outer_l0, kb_0, kb_iter_1_outer_l0, ob_0_out, p0_0, q_proj_tile_0, q_proj_tile_iter_1_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_1_aic(self, b_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 256], pl.FP32], gamma_iter_1_outer_l0: pl.Tensor[[1, 256], pl.FP32], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_0: pl.Tensor[[4, 1], pl.FP32], k0_2: pl.Scalar[pl.INDEX], k0_iter_6_outer_l0: pl.Scalar[pl.INDEX], k_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16], k_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 1024], pl.BFLOAT16], kb_2: pl.Scalar[pl.INDEX], kb_iter_4_outer_l0: pl.Scalar[pl.INDEX], normed_0: pl.Tensor[[4, 256], pl.FP32], normed_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32], ob_1_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], v_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16], v_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 1024], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], x_chunk_2: pl.Tensor[[4, 256], pl.FP32], x_chunk_iter_6_outer_l0: pl.Tensor[[4, 256], pl.FP32]) -> tuple[pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_6_outer_l1, k_proj_tile_iter_1_outer_l1, kb_iter_4_outer_l1, normed_iter_1_outer_l1, v_proj_tile_iter_1_outer_l1, x_chunk_iter_6_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_6_outer_l0, k_proj_tile_iter_1_outer_l0, kb_iter_4_outer_l0, normed_iter_1_outer_l0, v_proj_tile_iter_1_outer_l0, x_chunk_iter_6_outer_l0)):
            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 32
            for kb_6, (gamma_iter_3, k0_iter_8, normed_iter_3, x_chunk_iter_8) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_6_outer_l1, normed_iter_1_outer_l1, x_chunk_iter_6_outer_l1)):
                k0_10: pl.Scalar[pl.INDEX] = kb_6 * 256
                normed_bf16_0__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                normed_bf16_0__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                normed_bf16_0__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                normed_bf16_0__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(normed_bf16_0__tmp, normed_bf16_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                normed_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(normed_bf16_0__mid, normed_bf16_0__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                wk_chunk_0__h0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wk_chunk_0__h1: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wk_chunk_0__tmp: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.create(__list__(256, 32), dtype=pl.BFLOAT16)
                wk_chunk_0__mid: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wk_chunk_0__tmp, wk_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wk_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wk_chunk_0__mid, wk_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                wv_chunk_0__h0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wv_chunk_0__h1: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wv_chunk_0__tmp: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.create(__list__(256, 32), dtype=pl.BFLOAT16)
                wv_chunk_0__mid: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wv_chunk_0__tmp, wv_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wv_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wv_chunk_0__mid, wv_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t12: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t12, __list__(2, 32), __list__(0, 0))
                __half1__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t12, __list__(2, 32), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                _t13: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t13, __list__(2, 32), __list__(0, 0))
                __half1__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t13, __list__(2, 32), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                gamma_4, k0_9, normed_4, x_chunk_9 = pl.yield_(k0_10)
            gamma_iter_1_outer_l1_rv, k0_iter_6_outer_l1_rv, k_proj_tile_iter_1_outer_l1_rv, kb_iter_4_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_6_outer_l1_rv = pl.yield_(gamma_4, k0_9, kb_6, normed_4, x_chunk_9)
        return gamma_iter_1_outer_l1_rv, k0_iter_6_outer_l1_rv, k_proj_tile_iter_1_outer_l1_rv, kb_iter_4_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_6_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_1_aiv(self, b_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 256], pl.FP32], gamma_iter_1_outer_l0: pl.Tensor[[1, 256], pl.FP32], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_0: pl.Tensor[[4, 1], pl.FP32], k0_2: pl.Scalar[pl.INDEX], k0_iter_6_outer_l0: pl.Scalar[pl.INDEX], k_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16], k_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 1024], pl.BFLOAT16], kb_2: pl.Scalar[pl.INDEX], kb_iter_4_outer_l0: pl.Scalar[pl.INDEX], normed_0: pl.Tensor[[4, 256], pl.FP32], normed_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32], ob_1_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], v_proj_tile_0: pl.Tensor[[4, 1024], pl.BFLOAT16], v_proj_tile_iter_1_outer_l0: pl.Tensor[[4, 1024], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], x_chunk_2: pl.Tensor[[4, 256], pl.FP32], x_chunk_iter_6_outer_l0: pl.Tensor[[4, 256], pl.FP32], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[4, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.FP32]]:
        inv_rms_0_tile: pl.Tile[[4, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[4, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(inv_rms_0, [0, 0], [4, 1], [4, 1], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_6_outer_l1, k_proj_tile_iter_1_outer_l1, kb_iter_4_outer_l1, normed_iter_1_outer_l1, v_proj_tile_iter_1_outer_l1, x_chunk_iter_6_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_6_outer_l0, k_proj_tile_iter_1_outer_l0, kb_iter_4_outer_l0, normed_iter_1_outer_l0, v_proj_tile_iter_1_outer_l0, x_chunk_iter_6_outer_l0)):
            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 32
            k_acc_0: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.create([2, 32], dtype=pl.FP32)
            v_acc_0: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.create([2, 32], dtype=pl.FP32)
            k_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(k_acc_0, 0.0)
            v_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(v_acc_0, 0.0)
            for kb_6, (gamma_iter_3, k0_iter_8, k_acc_iter_2, normed_iter_3, v_acc_iter_2, x_chunk_iter_8) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_6_outer_l1, k_acc_1, normed_iter_1_outer_l1, v_acc_1, x_chunk_iter_6_outer_l1)):
                k0_10: pl.Scalar[pl.INDEX] = kb_6 * 256
                _t10: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 128], [b_0, p0_0 + AIV_IDX * 128, k0_10])
                x_chunk_10: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(_t10, target_type=pl.FP32, mode=2)
                gamma_5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_10 + AIV_IDX * 128])
                _t11: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_10, inv_rms_0)
                normed_5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t11, gamma_5)
                normed_bf16_0: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(normed_5, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(normed_bf16_0, AIV_IDX)
                wk_chunk_0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.tensor.view(wk_0, [128, 32], [k0_10 + AIV_IDX * 128, kv0_0])
                pl.comm.tpush_to_aic(wk_chunk_0, AIV_IDX)
                wv_chunk_0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.tensor.view(wv_0, [128, 32], [k0_10 + AIV_IDX * 128, kv0_0])
                pl.comm.tpush_to_aic(wv_chunk_0, AIV_IDX)
                _t12: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                k_acc_4: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.add(k_acc_iter_2, _t12)
                pl.comm.tfree_to_aic(AIV_IDX)
                _t13: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                v_acc_4: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.add(v_acc_iter_2, _t13)
                pl.comm.tfree_to_aic(AIV_IDX)
                gamma_4, k0_9, k_acc_3, normed_4, v_acc_3, x_chunk_9 = pl.yield_(gamma_5, k0_10, k_acc_4, normed_5, v_acc_4, x_chunk_10)
            _t14: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.cast(k_acc_3, target_type=pl.BFLOAT16, mode=2)
            k_proj_tile_3: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.assemble(k_proj_tile_iter_1_outer_l1, _t14, [0 + AIV_IDX * 2, kv0_0])
            _t15: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.cast(v_acc_3, target_type=pl.BFLOAT16, mode=2)
            v_proj_tile_3: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.assemble(v_proj_tile_iter_1_outer_l1, _t15, [0 + AIV_IDX * 2, kv0_0])
            gamma_iter_1_outer_l1_rv, k0_iter_6_outer_l1_rv, k_proj_tile_iter_1_outer_l1_rv, kb_iter_4_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_6_outer_l1_rv = pl.yield_(gamma_4, k0_9, k_proj_tile_3, kb_6, normed_4, v_proj_tile_3, x_chunk_9)
        return gamma_iter_1_outer_l1_rv, k0_iter_6_outer_l1_rv, k_proj_tile_iter_1_outer_l1_rv, kb_iter_4_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_tile_iter_1_outer_l1_rv, x_chunk_iter_6_outer_l1_rv
    @pl.function_group(aic="qwen3_prefill_layer_incore_1_aic", aiv="qwen3_prefill_layer_incore_1_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_prefill_layer_incore_1_group:
        """Parameter passing:
          call_group(qwen3_prefill_layer_incore_1_group, b_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_2, k0_iter_6_outer_l0, k_proj_tile_0, k_proj_tile_iter_1_outer_l0, kb_2, kb_iter_4_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, p0_0, v_proj_tile_0, v_proj_tile_iter_1_outer_l0, wk_0, wv_0, x_chunk_2, x_chunk_iter_6_outer_l0)
            → qwen3_prefill_layer_incore_1_aic(b_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_2, k0_iter_6_outer_l0, k_proj_tile_0, k_proj_tile_iter_1_outer_l0, kb_2, kb_iter_4_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, p0_0, v_proj_tile_0, v_proj_tile_iter_1_outer_l0, wk_0, wv_0, x_chunk_2, x_chunk_iter_6_outer_l0)
            → qwen3_prefill_layer_incore_1_aiv(b_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_0, k0_2, k0_iter_6_outer_l0, k_proj_tile_0, k_proj_tile_iter_1_outer_l0, kb_2, kb_iter_4_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, p0_0, v_proj_tile_0, v_proj_tile_iter_1_outer_l0, wk_0, wv_0, x_chunk_2, x_chunk_iter_6_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_2_aic(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_3: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_5: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_7_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16], k_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 1024], pl.BFLOAT16], pos_0: pl.Scalar[pl.INDEX], q_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 5120], pl.BFLOAT16], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ti_0: pl.Scalar[pl.INDEX], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_3: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_5: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_7_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 1024], pl.BFLOAT16]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Tensor[[524288, 128], pl.BFLOAT16], pl.Tensor[[524288, 128], pl.BFLOAT16]]:
        pl.comm.aic_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, k_cache_iter_7_outer_l1, v_cache_iter_7_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, k_cache_iter_7_outer_l0, v_cache_iter_7_outer_l0)):
            kvh_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            if (0 + (h_0_out * 8 + h_0_in) * 1) % 8 == 0:
                kv_col_0: pl.Scalar[pl.INDEX] = kvh_0 * 128
                cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + pos_0
            else:
                k_cache_10, v_cache_10 = pl.yield_(k_cache_iter_7_outer_l1, v_cache_iter_7_outer_l1)
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
            q_rot_bf16_0__discard: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
            pl.comm.tfree_to_aiv(1)
            for sb_0 in pl.range(0, ctx_blocks_0, 1):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                valid_len_0: pl.Scalar[pl.INDEX] = min(120, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + s0_0
                k_tile_0__h0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                k_tile_0__h1: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                k_tile_0__tmp: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.create(__list__(120, 128), dtype=pl.BFLOAT16)
                k_tile_0__mid: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(k_tile_0__tmp, k_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                k_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(k_tile_0__mid, k_tile_0__h1, __list__(60, 0))
                pl.comm.tfree_to_aiv(1)
                v_tile_0__h0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                v_tile_0__h1: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                v_tile_0__tmp: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.create(__list__(120, 128), dtype=pl.BFLOAT16)
                v_tile_0__mid: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(v_tile_0__tmp, v_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                v_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(v_tile_0__mid, v_tile_0__h1, __list__(60, 0))
                pl.comm.tfree_to_aiv(1)
                _t32: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                scores_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(_t32, 0.0883883)
                _t36: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.comm.tpop_from_aiv()
                oi_tmp_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(_t36, v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                pl.comm.tfree_to_aiv()
                if sb_0 == 0:
                    oi_4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp_0
                    li_6, mi_6, oi_6 = pl.yield_(oi_4)
                else:

                pl.yield_(li_6, mi_6, oi_6)
            pl.comm.tfree_to_aiv(0)
            attn_row_iter_2_outer_l1_rv, k_cache_iter_7_outer_l1_rv, v_cache_iter_7_outer_l1_rv = pl.yield_(k_cache_10, v_cache_10)
        return attn_row_iter_2_outer_l1_rv, k_cache_iter_7_outer_l1_rv, v_cache_iter_7_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_2_aiv(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_3: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_5: pl.Tensor[[524288, 128], pl.BFLOAT16], k_cache_iter_7_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16], k_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 1024], pl.BFLOAT16], pos_0: pl.Scalar[pl.INDEX], q_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 5120], pl.BFLOAT16], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ti_0: pl.Scalar[pl.INDEX], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_1: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_3: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_5: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_iter_7_outer_l0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_proj_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 1024], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Tensor[[524288, 128], pl.BFLOAT16], pl.Tensor[[524288, 128], pl.BFLOAT16]]:
        cos_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(cos_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        cos_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(cos_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_hi_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(sin_hi_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        sin_lo_0_tile: pl.Tile[[1, 128 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 128 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(sin_lo_0, [0, 0], [1, 128 // 2], [1, 128 // 2], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, k_cache_iter_7_outer_l1, v_cache_iter_7_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, k_cache_iter_7_outer_l0, v_cache_iter_7_outer_l0)):
            kvh_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            if (0 + (h_0_out * 8 + h_0_in) * 1) % 8 == 0:
                kv_col_0: pl.Scalar[pl.INDEX] = kvh_0 * 128
                _t16: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(k_proj_tile_iter_1_outer_l0_rv, [1, 64], [ti_0, kv_col_0 + AIV_IDX * 64])
                k_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t16, target_type=pl.FP32, mode=2)
                k_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(k_row_0, [1, 128 // 2], [0, 0])
                k_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(k_row_0, [1, 128 // 2], [0, 128 // 2])
                k_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                _t17: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, cos_lo_0)
                _t18: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, sin_lo_0)
                _t19: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t17, _t18)
                k_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_0, _t19, [0, 0])
                _t20: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, cos_hi_0)
                _t21: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, sin_hi_0)
                _t22: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t20, _t21)
                k_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_1, _t22, [0, 128 // 2])
                cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + pos_0
                _t23: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(k_rot_2, target_type=pl.BFLOAT16, mode=2)
                k_cache_9: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(k_cache_iter_7_outer_l1, _t23, [cache_row_0, 0])
                _t24: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(v_proj_tile_iter_1_outer_l0_rv, [1, 64], [ti_0, kv_col_0 + AIV_IDX * 64])
                v_cache_9: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(v_cache_iter_7_outer_l1, _t24, [cache_row_0 + AIV_IDX * 64, 0])
                k_cache_10, v_cache_10 = pl.yield_(k_cache_9, v_cache_9)
            else:
                k_cache_10, v_cache_10 = pl.yield_(k_cache_iter_7_outer_l1, v_cache_iter_7_outer_l1)
            _t25: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_l0_rv, [1, 64], [ti_0, q_col_0 + AIV_IDX * 64])
            q_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t25, target_type=pl.FP32, mode=2)
            q_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 0])
            q_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 128 // 2])
            q_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
            _t26: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
            _t27: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
            _t28: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t26, _t27)
            q_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_0, _t28, [0, 0])
            _t29: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
            _t30: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
            _t31: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t29, _t30)
            q_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_1, _t31, [0, 128 // 2])
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_rot_2, target_type=pl.BFLOAT16, mode=2)
            pl.comm.tpush_to_aic(q_rot_bf16_0, AIV_IDX)
            oi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
            li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(oi_0, 0.0)
            li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li_0, 0.0)
            mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi_0, 0.0)
            for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                valid_len_0: pl.Scalar[pl.INDEX] = min(120, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + s0_0
                k_tile_0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.tensor.deep_view(k_cache_10, [60, 128], [cache_row0_0, 0])
                pl.comm.tpush_to_aic(k_tile_0, AIV_IDX)
                v_tile_0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.tensor.deep_view(v_cache_10, [60, 128], [cache_row0_0, 0])
                pl.comm.tpush_to_aic(v_tile_0, AIV_IDX)
                exp_pad_0: pl.Tensor[[1, 60], pl.FP32] = pl.tensor.create([1, 60], dtype=pl.FP32)
                exp_pad_1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(exp_pad_0, 0.0)
            ctx_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
            attn_row_4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_iter_2_outer_l1, ctx_0, [0, q_col_0])
            attn_row_iter_2_outer_l1_rv, k_cache_iter_7_outer_l1_rv, v_cache_iter_7_outer_l1_rv = pl.yield_(attn_row_4, k_cache_10, v_cache_10)
        return attn_row_iter_2_outer_l1_rv, k_cache_iter_7_outer_l1_rv, v_cache_iter_7_outer_l1_rv
    @pl.function_group(aic="qwen3_prefill_layer_incore_2_aic", aiv="qwen3_prefill_layer_incore_2_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_prefill_layer_incore_2_group:
        """Parameter passing:
          call_group(qwen3_prefill_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_0, k_cache_iter_1, k_cache_iter_3, k_cache_iter_5, k_cache_iter_7_outer_l0, k_proj_tile_iter_1_outer_l0_rv, pos_0, q_proj_tile_iter_1_outer_l0_rv, sin_hi_0, sin_lo_0, ti_0, v_cache_0, v_cache_iter_1, v_cache_iter_3, v_cache_iter_5, v_cache_iter_7_outer_l0, v_proj_tile_iter_1_outer_l0_rv)
            → qwen3_prefill_layer_incore_2_aic(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_0, k_cache_iter_1, k_cache_iter_3, k_cache_iter_5, k_cache_iter_7_outer_l0, k_proj_tile_iter_1_outer_l0_rv, pos_0, q_proj_tile_iter_1_outer_l0_rv, sin_hi_0, sin_lo_0, ti_0, v_cache_0, v_cache_iter_1, v_cache_iter_3, v_cache_iter_5, v_cache_iter_7_outer_l0, v_proj_tile_iter_1_outer_l0_rv)
            → qwen3_prefill_layer_incore_2_aiv(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_0, k_cache_iter_1, k_cache_iter_3, k_cache_iter_5, k_cache_iter_7_outer_l0, k_proj_tile_iter_1_outer_l0_rv, pos_0, q_proj_tile_iter_1_outer_l0_rv, sin_hi_0, sin_lo_0, ti_0, v_cache_0, v_cache_iter_1, v_cache_iter_3, v_cache_iter_5, v_cache_iter_7_outer_l0, v_proj_tile_iter_1_outer_l0_rv, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_3_aic(self, attn_tile_3: pl.Tensor[[4, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], k0_7: pl.Scalar[pl.INDEX], k0_iter_11_outer_l0: pl.Scalar[pl.INDEX], kb_5: pl.Scalar[pl.INDEX], kb_iter_7_outer_l0: pl.Scalar[pl.INDEX], ob_2_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.FP32], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_2_in, (k0_iter_11_outer_l1, kb_iter_7_outer_l1, resid1_tile_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_11_outer_l0, kb_iter_7_outer_l0, resid1_tile_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_2_out * 8 + ob_2_in) * 1) * 64
            for kb_9, (k0_iter_13,) in pl.range(0, 20, 1, init_values=(k0_iter_11_outer_l1,)):
                k0_15: pl.Scalar[pl.INDEX] = kb_9 * 256
                a_chunk_0__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                a_chunk_0__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                a_chunk_0__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                a_chunk_0__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(a_chunk_0__tmp, a_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                a_chunk_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(a_chunk_0__mid, a_chunk_0__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                w_chunk_0__h0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                w_chunk_0__h1: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                w_chunk_0__tmp: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.create(__list__(256, 64), dtype=pl.BFLOAT16)
                w_chunk_0__mid: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_chunk_0__tmp, w_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                w_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_chunk_0__mid, w_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t44: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t44, __list__(2, 64), __list__(0, 0))
                __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t44, __list__(2, 64), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_14: pl.Scalar[pl.INDEX] = pl.yield_(k0_15)
            k0_iter_11_outer_l1_rv, kb_iter_7_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv = pl.yield_(k0_14, kb_9)
        return k0_iter_11_outer_l1_rv, kb_iter_7_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_3_aiv(self, attn_tile_3: pl.Tensor[[4, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], k0_7: pl.Scalar[pl.INDEX], k0_iter_11_outer_l0: pl.Scalar[pl.INDEX], kb_5: pl.Scalar[pl.INDEX], kb_iter_7_outer_l0: pl.Scalar[pl.INDEX], ob_2_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.FP32], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_2_in, (k0_iter_11_outer_l1, kb_iter_7_outer_l1, resid1_tile_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_11_outer_l0, kb_iter_7_outer_l0, resid1_tile_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_2_out * 8 + ob_2_in) * 1) * 64
            o_acc_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.create([2, 64], dtype=pl.FP32)
            o_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
            for kb_9, (k0_iter_13, o_acc_iter_2) in pl.range(0, 20, 1, init_values=(k0_iter_11_outer_l1, o_acc_1)):
                k0_15: pl.Scalar[pl.INDEX] = kb_9 * 256
                _t43: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.view(attn_tile_3, [2, 256], [0 + AIV_IDX * 2, k0_15])
                a_chunk_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(_t43, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(a_chunk_0, AIV_IDX)
                w_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wo_0, [128, 64], [k0_15 + AIV_IDX * 128, o0_0])
                pl.comm.tpush_to_aic(w_chunk_0, AIV_IDX)
                _t44: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                o_acc_4: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(o_acc_iter_2, _t44)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_14, o_acc_3 = pl.yield_(k0_15, o_acc_4)
            _t45: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [2, 64], [b_0 + AIV_IDX * 2, p0_0, o0_0])
            resid_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(_t45, target_type=pl.FP32, mode=2)
            _t46: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(o_acc_3, resid_0)
            resid1_tile_3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1_outer_l1, _t46, [0 + AIV_IDX * 2, o0_0])
            k0_iter_11_outer_l1_rv, kb_iter_7_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv = pl.yield_(k0_14, kb_9, resid1_tile_3)
        return k0_iter_11_outer_l1_rv, kb_iter_7_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_prefill_layer_incore_3_aic", aiv="qwen3_prefill_layer_incore_3_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_prefill_layer_incore_3_group:
        """Parameter passing:
          call_group(qwen3_prefill_layer_incore_3_group, attn_tile_3, b_0, hidden_states_0, k0_7, k0_iter_11_outer_l0, kb_5, kb_iter_7_outer_l0, ob_2_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → qwen3_prefill_layer_incore_3_aic(attn_tile_3, b_0, hidden_states_0, k0_7, k0_iter_11_outer_l0, kb_5, kb_iter_7_outer_l0, ob_2_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → qwen3_prefill_layer_incore_3_aiv(attn_tile_3, b_0, hidden_states_0, k0_7, k0_iter_11_outer_l0, kb_5, kb_iter_7_outer_l0, ob_2_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_4_aic(self, b_0: pl.Scalar[pl.INDEX], dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_2: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 5120], pl.FP32], mlp_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16], o0_3: pl.Scalar[pl.INDEX], ob_3: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_1: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_3: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_5: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_7_outer_l0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], p0_0: pl.Scalar[pl.INDEX], resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 5120], pl.FP32], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16]) -> tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]]:
        pl.comm.aic_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1, out_iter_7_outer_l1) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer_l0, out_iter_7_outer_l0)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
            w_down_chunk_0__h0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
            w_down_chunk_0__h1: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
            w_down_chunk_0__tmp: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.create(__list__(256, 64), dtype=pl.BFLOAT16)
            w_down_chunk_0__mid: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_down_chunk_0__tmp, w_down_chunk_0__h0, __list__(0, 0))
            pl.comm.tfree_to_aiv(0)
            w_down_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_down_chunk_0__mid, w_down_chunk_0__h1, __list__(128, 0))
            pl.comm.tfree_to_aiv(1)
            _t59: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t59, __list__(2, 64), __list__(0, 0))
            __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t59, __list__(2, 64), __list__(2, 0))
            pl.comm.tpush_to_aiv(__half0__, 0)
            pl.comm.tpush_to_aiv(__half1__, 1)
            if ob_3 == 100 - 1:

            else:
                out_10: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.yield_(out_iter_7_outer_l1)
            down_proj_tile_iter_4_outer_l1_rv, out_iter_7_outer_l1_rv = pl.yield_(out_10)
        return down_proj_tile_iter_4_outer_l1_rv, out_iter_7_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_4_aiv(self, b_0: pl.Scalar[pl.INDEX], dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_2: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 5120], pl.FP32], mlp_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16], o0_3: pl.Scalar[pl.INDEX], ob_3: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_1: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_3: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_5: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], out_iter_7_outer_l0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], p0_0: pl.Scalar[pl.INDEX], resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 5120], pl.FP32], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]]:
        pl.comm.aiv_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1, out_iter_7_outer_l1) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer_l0, out_iter_7_outer_l0)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
            down_prev_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(down_proj_tile_iter_4_outer_l1, [2, 64], [0, d0_0])
            w_down_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(w_down_0, [128, 64], [o0_3 + AIV_IDX * 128, d0_0])
            pl.comm.tpush_to_aic(w_down_chunk_0, AIV_IDX)
            _t59: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
            down_next_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(down_prev_0, _t59)
            pl.comm.tfree_to_aic(AIV_IDX)
            down_proj_tile_6: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile_iter_4_outer_l1, down_next_0, [0 + AIV_IDX * 2, d0_0])
            if ob_3 == 100 - 1:
                _t60: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.deep_view(down_proj_tile_6, [2, 64], [0, d0_0])
                _t61: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [2, 64], [0 + AIV_IDX * 2, d0_0])
                down_acc_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(_t60, _t61)
                _t62: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
                out_9: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.tensor.assemble(out_iter_7_outer_l1, _t62, [b_0 + AIV_IDX * 2, p0_0, d0_0])
                out_10: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.yield_(out_9)
            else:
                out_10: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.yield_(out_iter_7_outer_l1)
            down_proj_tile_iter_4_outer_l1_rv, out_iter_7_outer_l1_rv = pl.yield_(down_proj_tile_6, out_10)
        return down_proj_tile_iter_4_outer_l1_rv, out_iter_7_outer_l1_rv
    @pl.function_group(aic="qwen3_prefill_layer_incore_4_aic", aiv="qwen3_prefill_layer_incore_4_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_prefill_layer_incore_4_group:
        """Parameter passing:
          call_group(qwen3_prefill_layer_incore_4_group, b_0, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, ob_3, out_0, out_iter_1, out_iter_3, out_iter_5, out_iter_7_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv, w_down_0)
            → qwen3_prefill_layer_incore_4_aic(b_0, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, ob_3, out_0, out_iter_1, out_iter_3, out_iter_5, out_iter_7_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv, w_down_0)
            → qwen3_prefill_layer_incore_4_aiv(b_0, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, ob_3, out_0, out_iter_1, out_iter_3, out_iter_5, out_iter_7_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv, w_down_0, AIV_IDX=<runtime>)
        """
        pass
