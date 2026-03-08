# pypto.program: Qwen3SingleLayerPrefill
import pypto.language as pl

@pl.program
class Qwen3SingleLayerPrefill:
    @pl.function
    def qwen3_prefill_layer(self, hidden_states_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], out_0: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]) -> pl.Tensor[[16, 4096, 5120], pl.BFLOAT16]:
        for b_0, (k_cache_iter_1, out_iter_1, v_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(k_cache_0, out_0, v_cache_0), chunk=4):
            for p0_0, (k_cache_iter_3, out_iter_3, v_cache_iter_3) in pl.range(0, 4096, 4, init_values=(k_cache_iter_1, out_iter_1, v_cache_iter_1)):
                with pl.auto_incore():
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
                    for ob_0_out, (k0_iter_1_outer, kb_iter_1_outer, q_proj_tile_iter_1_outer, x_chunk_iter_1_outer) in pl.range(0, 10, 1, init_values=(k0_0, kb_0, q_proj_tile_0, x_chunk_0)):
                        for ob_0_in, (k0_iter_1_inner, kb_iter_1_inner, q_proj_tile_iter_1_inner, x_chunk_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(k0_iter_1_outer, kb_iter_1_outer, q_proj_tile_iter_1_outer, x_chunk_iter_1_outer)):
                            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 64
                            q_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32)
                            q_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(q_acc_0, 0.0)
                            for kb_3, (k0_iter_3, q_acc_iter_2, x_chunk_iter_3) in pl.range(0, 20, 1, init_values=(k0_iter_1_inner, q_acc_1, x_chunk_iter_1_inner)):
                                k0_5: pl.Scalar[pl.INDEX] = kb_3 * 256
                                _t5: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 256], [b_0, p0_0, k0_5])
                                x_chunk_5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(_t5, target_type=pl.FP32, mode=2)
                                gamma_0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 256], [0, k0_5])
                                _t6: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_5, inv_rms_0)
                                normed_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t6, gamma_0)
                                wq_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.view(wq_0, [256, 64], [k0_5, q0_0])
                                _t7: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                                _t8: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(_t7, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                q_acc_4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc_iter_2, _t8)
                                k0_4, q_acc_3, x_chunk_4 = pl.yield_(k0_5, q_acc_4, x_chunk_5)
                            _t9: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2)
                            q_proj_tile_3: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile_iter_1_inner, _t9, [0, q0_0])
                            k0_iter_1_inner_rv, kb_iter_1_inner_rv, q_proj_tile_iter_1_inner_rv, x_chunk_iter_1_inner_rv = pl.yield_(k0_4, kb_3, q_proj_tile_3, x_chunk_4)
                        k0_iter_1_outer_rv, kb_iter_1_outer_rv, q_proj_tile_iter_1_outer_rv, x_chunk_iter_1_outer_rv = pl.yield_(k0_iter_1_inner_rv, kb_iter_1_inner_rv, q_proj_tile_iter_1_inner_rv, x_chunk_iter_1_inner_rv)
                    for ob_1_out, (gamma_iter_1_outer, k0_iter_6_outer, k_proj_tile_iter_1_outer, kb_iter_4_outer, normed_iter_1_outer, v_proj_tile_iter_1_outer, x_chunk_iter_6_outer) in pl.range(0, 4, 1, init_values=(gamma_0, k0_2, k_proj_tile_0, kb_2, normed_0, v_proj_tile_0, x_chunk_2)):
                        for ob_1_in, (gamma_iter_1_inner, k0_iter_6_inner, k_proj_tile_iter_1_inner, kb_iter_4_inner, normed_iter_1_inner, v_proj_tile_iter_1_inner, x_chunk_iter_6_inner) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer, k0_iter_6_outer, k_proj_tile_iter_1_outer, kb_iter_4_outer, normed_iter_1_outer, v_proj_tile_iter_1_outer, x_chunk_iter_6_outer)):
                            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 32
                            k_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                            v_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                            k_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(k_acc_0, 0.0)
                            v_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(v_acc_0, 0.0)
                            for kb_6, (gamma_iter_3, k0_iter_8, k_acc_iter_2, normed_iter_3, v_acc_iter_2, x_chunk_iter_8) in pl.range(0, 20, 1, init_values=(gamma_iter_1_inner, k0_iter_6_inner, k_acc_1, normed_iter_1_inner, v_acc_1, x_chunk_iter_6_inner)):
                                k0_10: pl.Scalar[pl.INDEX] = kb_6 * 256
                                _t10: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 256], [b_0, p0_0, k0_10])
                                x_chunk_10: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(_t10, target_type=pl.FP32, mode=2)
                                gamma_5: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 256], [0, k0_10])
                                _t11: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_10, inv_rms_0)
                                normed_5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t11, gamma_5)
                                normed_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(normed_5, target_type=pl.BFLOAT16, mode=2)
                                wk_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(wk_0, [256, 32], [k0_10, kv0_0])
                                wv_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(wv_0, [256, 32], [k0_10, kv0_0])
                                _t12: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                k_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc_iter_2, _t12)
                                _t13: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                v_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc_iter_2, _t13)
                                gamma_4, k0_9, k_acc_3, normed_4, v_acc_3, x_chunk_9 = pl.yield_(gamma_5, k0_10, k_acc_4, normed_5, v_acc_4, x_chunk_10)
                            _t14: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.cast(k_acc_3, target_type=pl.BFLOAT16, mode=2)
                            k_proj_tile_3: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.assemble(k_proj_tile_iter_1_inner, _t14, [0, kv0_0])
                            _t15: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.cast(v_acc_3, target_type=pl.BFLOAT16, mode=2)
                            v_proj_tile_3: pl.Tensor[[4, 1024], pl.BFLOAT16] = pl.tensor.assemble(v_proj_tile_iter_1_inner, _t15, [0, kv0_0])
                            gamma_iter_1_inner_rv, k0_iter_6_inner_rv, k_proj_tile_iter_1_inner_rv, kb_iter_4_inner_rv, normed_iter_1_inner_rv, v_proj_tile_iter_1_inner_rv, x_chunk_iter_6_inner_rv = pl.yield_(gamma_4, k0_9, k_proj_tile_3, kb_6, normed_4, v_proj_tile_3, x_chunk_9)
                        gamma_iter_1_outer_rv, k0_iter_6_outer_rv, k_proj_tile_iter_1_outer_rv, kb_iter_4_outer_rv, normed_iter_1_outer_rv, v_proj_tile_iter_1_outer_rv, x_chunk_iter_6_outer_rv = pl.yield_(gamma_iter_1_inner_rv, k0_iter_6_inner_rv, k_proj_tile_iter_1_inner_rv, kb_iter_4_inner_rv, normed_iter_1_inner_rv, v_proj_tile_iter_1_inner_rv, x_chunk_iter_6_inner_rv)
                with pl.auto_incore():
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
                        for h_0_out, (attn_row_iter_2_outer, k_cache_iter_7_outer, v_cache_iter_7_outer) in pl.range(0, 8, 1, init_values=(attn_row_1, k_cache_iter_5, v_cache_iter_5)):
                            for h_0_in, (attn_row_iter_2_inner, k_cache_iter_7_inner, v_cache_iter_7_inner) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer, k_cache_iter_7_outer, v_cache_iter_7_outer)):
                                kvh_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
                                q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
                                if (0 + (h_0_out * 8 + h_0_in) * 1) % 8 == 0:
                                    kv_col_0: pl.Scalar[pl.INDEX] = kvh_0 * 128
                                    _t16: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(k_proj_tile_iter_1_outer_rv, [1, 128], [ti_0, kv_col_0])
                                    k_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t16, target_type=pl.FP32, mode=2)
                                    k_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 0])
                                    k_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 128 // 2])
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
                                    k_cache_9: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(k_cache_iter_7_inner, _t23, [cache_row_0, 0])
                                    _t24: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(v_proj_tile_iter_1_outer_rv, [1, 128], [ti_0, kv_col_0])
                                    v_cache_9: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(v_cache_iter_7_inner, _t24, [cache_row_0, 0])
                                    k_cache_10, v_cache_10 = pl.yield_(k_cache_9, v_cache_9)
                                else:
                                    k_cache_10, v_cache_10 = pl.yield_(k_cache_iter_7_inner, v_cache_iter_7_inner)
                                _t25: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_rv, [1, 128], [ti_0, q_col_0])
                                q_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t25, target_type=pl.FP32, mode=2)
                                q_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row_0, [1, 128 // 2], [0, 0])
                                q_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row_0, [1, 128 // 2], [0, 128 // 2])
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
                                oi_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                                li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(oi_0, 0.0)
                                li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li_0, 0.0)
                                mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi_0, 0.0)
                                for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
                                    s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                                    valid_len_0: pl.Scalar[pl.INDEX] = min(120, ctx_len_0 - s0_0)
                                    cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + s0_0
                                    k_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.view(k_cache_10, [120, 128], [cache_row0_0, 0])
                                    v_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.view(v_cache_10, [120, 128], [cache_row0_0, 0])
                                    _t32: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                                    scores_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(_t32, 0.0883883)
                                    scores_valid_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.view(scores_0, [1, valid_len_0], [0, 0])
                                    _t33: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_max(scores_valid_0)
                                    cur_mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(_t33, target_type=pl.FP32, mode=2)
                                    _t34: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.row_expand_sub(scores_valid_0, cur_mi_0)
                                    exp_scores_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.exp(_t34)
                                    _t35: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(exp_scores_0)
                                    cur_li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(_t35, target_type=pl.FP32, mode=2)
                                    exp_pad_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32)
                                    exp_pad_1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(exp_pad_0, 0.0)
                                    exp_pad_2: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad_1, exp_scores_0, [0, 0])
                                    _t36: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.tensor.cast(exp_pad_2, target_type=pl.BFLOAT16, mode=2)
                                    oi_tmp_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(_t36, v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                    if sb_0 == 0:
                                        oi_4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp_0
                                        li_4: pl.Tensor[[1, 1], pl.FP32] = cur_li_0
                                        mi_4: pl.Tensor[[1, 1], pl.FP32] = cur_mi_0
                                        li_6, mi_6, oi_6 = pl.yield_(li_4, mi_4, oi_4)
                                    else:
                                        mi_new_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi_iter_2, cur_mi_0)
                                        _t37: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(mi_iter_2, mi_new_0)
                                        alpha_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t37)
                                        _t38: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(cur_mi_0, mi_new_0)
                                        beta_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t38)
                                        _t39: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(alpha_0, li_iter_2)
                                        _t40: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(beta_0, cur_li_0)
                                        li_5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(_t39, _t40)
                                        _t41: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi_iter_2, alpha_0)
                                        _t42: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi_tmp_0, beta_0)
                                        oi_5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(_t41, _t42)
                                        mi_5: pl.Tensor[[1, 1], pl.FP32] = mi_new_0
                                        li_6, mi_6, oi_6 = pl.yield_(li_5, mi_5, oi_5)
                                    li_3, mi_3, oi_3 = pl.yield_(li_6, mi_6, oi_6)
                                ctx_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
                                attn_row_4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_iter_2_inner, ctx_0, [0, q_col_0])
                                attn_row_iter_2_inner_rv, k_cache_iter_7_inner_rv, v_cache_iter_7_inner_rv = pl.yield_(attn_row_4, k_cache_10, v_cache_10)
                            attn_row_iter_2_outer_rv, k_cache_iter_7_outer_rv, v_cache_iter_7_outer_rv = pl.yield_(attn_row_iter_2_inner_rv, k_cache_iter_7_inner_rv, v_cache_iter_7_inner_rv)
                        attn_tile_4: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(attn_tile_iter_2, attn_row_iter_2_outer_rv, [ti_0, 0])
                        attn_tile_3, k_cache_6, v_cache_6 = pl.yield_(attn_tile_4, k_cache_iter_7_outer_rv, v_cache_iter_7_outer_rv)
                with pl.auto_incore():
                    resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                    for ob_2_out, (k0_iter_11_outer, kb_iter_7_outer, resid1_tile_iter_1_outer) in pl.range(0, 10, 1, init_values=(k0_7, kb_5, resid1_tile_0)):
                        for ob_2_in, (k0_iter_11_inner, kb_iter_7_inner, resid1_tile_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(k0_iter_11_outer, kb_iter_7_outer, resid1_tile_iter_1_outer)):
                            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_2_out * 8 + ob_2_in) * 1) * 64
                            o_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32)
                            o_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
                            for kb_9, (k0_iter_13, o_acc_iter_2) in pl.range(0, 20, 1, init_values=(k0_iter_11_inner, o_acc_1)):
                                k0_15: pl.Scalar[pl.INDEX] = kb_9 * 256
                                _t43: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(attn_tile_3, [4, 256], [0, k0_15])
                                a_chunk_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(_t43, target_type=pl.BFLOAT16, mode=2)
                                w_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.view(wo_0, [256, 64], [k0_15, o0_0])
                                _t44: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                o_acc_4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc_iter_2, _t44)
                                k0_14, o_acc_3 = pl.yield_(k0_15, o_acc_4)
                            _t45: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 64], [b_0, p0_0, o0_0])
                            resid_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(_t45, target_type=pl.FP32, mode=2)
                            _t46: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc_3, resid_0)
                            resid1_tile_3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1_inner, _t46, [0, o0_0])
                            k0_iter_11_inner_rv, kb_iter_7_inner_rv, resid1_tile_iter_1_inner_rv = pl.yield_(k0_14, kb_9, resid1_tile_3)
                        k0_iter_11_outer_rv, kb_iter_7_outer_rv, resid1_tile_iter_1_outer_rv = pl.yield_(k0_iter_11_inner_rv, kb_iter_7_inner_rv, resid1_tile_iter_1_inner_rv)
                    sq_sum_5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                    sq_sum_6: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_5, 0.0)
                    for kb_10, (k0_iter_16, sq_sum_iter_7, x_chunk_iter_11) in pl.range(0, 20, 1, init_values=(k0_iter_11_outer_rv, sq_sum_6, x_chunk_iter_6_outer_rv)):
                        k0_18: pl.Scalar[pl.INDEX] = kb_10 * 256
                        x_chunk_13: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_rv, [4, 256], [0, k0_18])
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
                    for kb_11, (gamma_iter_6, k0_iter_19, normed_iter_6, post_norm_tile_iter_1, x_chunk_iter_14) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_rv, k0_17, normed_iter_1_outer_rv, post_norm_tile_0, x_chunk_12)):
                        k0_21: pl.Scalar[pl.INDEX] = kb_11 * 256
                        x_chunk_16: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_rv, [4, 256], [0, k0_21])
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
                        for dob_0_out, (down_proj_tile_iter_4_outer, out_iter_7_outer) in pl.range(0, 10, 1, init_values=(down_proj_tile_iter_2, out_iter_5)):
                            for dob_0_in, (down_proj_tile_iter_4_inner, out_iter_7_inner) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer, out_iter_7_outer)):
                                d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 64
                                down_prev_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(down_proj_tile_iter_4_inner, [4, 64], [0, d0_0])
                                w_down_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.view(w_down_0, [256, 64], [o0_3, d0_0])
                                _t59: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                down_next_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev_0, _t59)
                                down_proj_tile_6: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile_iter_4_inner, down_next_0, [0, d0_0])
                                if ob_3 == 100 - 1:
                                    _t60: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(down_proj_tile_6, [4, 64], [0, d0_0])
                                    _t61: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_rv, [4, 64], [0, d0_0])
                                    down_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(_t60, _t61)
                                    _t62: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
                                    out_9: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.tensor.assemble(out_iter_7_inner, _t62, [b_0, p0_0, d0_0])
                                    out_10: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.yield_(out_9)
                                else:
                                    out_10: pl.Tensor[[16, 4096, 5120], pl.BFLOAT16] = pl.yield_(out_iter_7_inner)
                                down_proj_tile_iter_4_inner_rv, out_iter_7_inner_rv = pl.yield_(down_proj_tile_6, out_10)
                            down_proj_tile_iter_4_outer_rv, out_iter_7_outer_rv = pl.yield_(down_proj_tile_iter_4_inner_rv, out_iter_7_inner_rv)
                        down_proj_tile_3, k0_23, kb_13, o0_2, out_6 = pl.yield_(down_proj_tile_iter_4_outer_rv, k0_25, kb_14, o0_3, out_iter_7_outer_rv)
                k_cache_4, out_4, v_cache_4 = pl.yield_(k_cache_6, out_6, v_cache_6)
            k_cache_2, out_2, v_cache_2 = pl.yield_(k_cache_4, out_4, v_cache_4)
        return out_2