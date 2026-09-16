# 9月1日 RMSNorm 学习讨论记录

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/9月1日RMSNorm 学习讨论记录.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

讨论文件：`models/deepseek_v4_flash_mtp/rmsnorm.py`

1. `T_DYN` 是什么——表示动态的 token 数，`T=B×S`。decode 的 `T=8`，prefill 的 `T=128`；`D=4096` 和 `EPS=1e-6` 是固定配置。

2. 为什么这么切分，`D_TILE=128` 和 `T_TILE=8` 是怎么定的——没有看到额外的硬约束，代码里的约束主要是 28～30 行的三个 `assert`，保证 `D`、decode 的 `T` 和 prefill 的 `T` 都能整除。具体取值还是 tiling，需要结合性能去调。

3. 是不是每个核只处理 128 个元素——不是。`T_TILE=8` 表示一个逻辑 block 处理 8 个 token；`D_TILE=128` 是这个 block 内部每次处理 128 个 hidden 元素。每个 block 最后还是要把完整的 `D=4096` 处理完，一共循环 32 次。

4. `t_dim = pl.tensor.dim(x, 0)` 获取的是不是核数——不是，获取的是第 0 维大小，也就是 token 数。真正的逻辑 block 数是 `t_dim // T_TILE`，再由运行时分配到物理核。

5. inline 是不是每个函数都要加，函数调用有深度要求吗——看怎么用。单测如果直接写实现可以不用；其他 JIT 需要复用这个函数时可以写成 inline。函数调用没有深度限制，但不支持递归调用。

6. `pl.spmd(t_dim // T_TILE)` 怎么调，并行度是不是越高越好——不一定。block 少了可能用不满核，block 太多、每块工作太小也可能增加开销，还是要结合 `T_TILE`、`D_TILE` 和实际场景一起调。

7. `allow_early_resolve=True` 是什么作用——MPMD 场景由 AICPU 做调度，这个标志加在上游 producer 上，让下游 consumer 有机会提前下发并等待上游完成。先看关键路径和调度空隙，只在确实需要的链路上加，不建议到处加。

8. `x_sq_sum = pl.full(..., value=0.0)` 是不是 UB 上的数据，会不会真的清零——是 UB 上的 Vec tile，大小为 `[1,8]`，用于保存 8 个 token 的 FP32 平方和。编译器会给它分配 UB 地址，并生成 `TEXPANDS(..., 0.0)`，实际写入 8 个 FP32 的 0。这里初始化的只是 `x_sq_sum` 这块有效数据，不是把整个 UB 清零。

9. `pl.pipeline(D // D_TILE, stage=2)` 里的 `stage=2` 是什么——软流水是把不同循环轮次的“搬运”和“计算”交错起来执行。普通循环是第 0 块完成 load、cast、计算以后，才开始处理第 1 块；软流水会在计算第 0 块时，提前搬运第 1 块，尽量让 MTE 搬运和 Vector 计算同时工作。

   `stage=2` 表示同时准备两轮数据，也就是 double buffer：第 0 块使用 buffer A 计算时，第 1 块可以加载到 buffer B；第 1 块开始计算后，buffer A 再用于下一块。两个 buffer 轮换，可以避免下一块 load 覆盖当前还在计算的数据。

   ```text
   MTE2：load 0(A) | load 1(B) | load 2(A) | ...
   Vector：          compute 0(A) | compute 1(B) | ...
   ```

   在这个 RMSNorm 里，一轮就是一个 `[8,128]` 的 D 维 chunk。现有 `.pto` 中原来的 32 轮循环被改成 step 2，并在循环体里生成两轮相邻 chunk 的处理代码。`stage=2` 不表示使用两个核，而是让同一个 kernel 内最多有两轮数据处在流水中。实际能否完全重叠，还要看数据依赖、UB 空间和生成指令。stage 要大于等于 1，项目中常见为 2～4；越大需要的片上 buffer 和代码也越多。

10. `pl.cast` 为什么不放到 `for` 循环外面——完整 `[8,4096]` 的 FP32 `x` 就有 128 KiB，外提后 UB 压力会明显增加；现在每次只 cast `[8,128]`，FP32 只有 4 KiB。是否真的会爆 UB、性能怎样，待验证。

11. `rms_x_chunk` 的 UB 地址什么时候分配，会不会有内存碎片——地址由编译器的内存复用和地址分配阶段确定。现有 `.pto` 中，`rms_x_chunk` 的 UB offset 是 8736，shape 为 `[8,128]` FP32，占 4096B，对应区间 `[8736,12832)`；后面的 `mul` 结果继续使用 offset 8736。`stage=2` 的另一轮 FP32 chunk 使用 offset 0，归约临时区会使用 4096 和 12832 等位置。当前 Vec 地址最高使用到约 16928B，0、4096、8192、8224、8480、8736、12832 等位置被不同生命周期的 tile 反复复用。从这份 `.pto` 看不出明显的内存碎片，部分小间隔主要和对齐、tile 大小及生命周期有关；如果要看完整占用比例，可以再用 memory map 查看。

12. `pl.mul(rms_x_chunk, rms_x_chunk)` 是不是原地累乘——前端语义不是原地修改，会产生一个新结果；但当前生成 IR 里输入和输出复用了同一段 Vec 地址，所以底层看起来是原地式复用。这是编译器分配结果，不是 `pl.mul` 的固定语义。

13. 两个输入一样会不会发生 UB bank conflict——只能确认生成代码是 `TMUL(dst, src, src)`，不能直接确认有 bank conflict。UB 地址由框架分配，需要继续结合分配后的 IR 和核内分析看，待验证。

14. `row_sum` 是 reduce、add 拼接，还是别的方案——当前直接生成专用的 `TROWSUM`，不是在 PyPTO 层手工写一串 add。具体底层怎么归约、用了多少指令，由 PTO ISA 根据 tile shape 决定。

15. `reshape` 会不会做 transpose 或搬数据——当前 `[8,1]` reshape 成 `[1,8]`，生成 IR 中前后使用同一个地址，没有数据搬运，只是换一种 shape 解释。但不能把这个结论套到所有 reshape 上。

16. `1/sqrt` 和 `rsqrt` 哪个更快，前端写法会不会影响 CCE 指令——当前写的是 `rsqrt(high_precision=True)`，生成 `TRSQRT`。在当前 a2a3 高精度实现中，底层是 `vsqrt + vdiv`；非高精度路径可以走 `vrsqrt`。不同写法和精度选项会影响最终指令，性能差异待验证。

17. 第一遍已经 cast 过 `x`，为什么第二遍还要再 cast——第一遍只保留当前 `[8,128]` chunk，用完就释放，没有保存完整 FP32 `x`。所以第二遍需要重新 load 和 cast，这是用重复转换换较小的 UB 占用。

18. `norm_w` 能不能提前整体 cast——完整 FP32 `norm_w` 只有 16 KiB，比完整 FP32 `x` 小很多，可以单独尝试外提；是否更快、是否影响流水，待验证。

19. `row_expand_mul`、`col_expand_mul` 这种复合操作和分开写哪个更快——当前分别生成专用的 `TROWEXPANDMUL` 和 `TCOLEXPANDMUL`，不会先把 broadcast 数据完整展开。和其他写法的性能差异还是要看生成代码和实测。

20. `x[...]` 这种切片是跳 stride，还是先拼成连续数据——生成的是带 offset、size 和 stride 的 view。每行 128 个元素连续，不同行之间按原始 `D=4096` 跳 stride，再由 `TLOAD` 搬到 Vec tile，不会先在 GM 中拼一个新 tensor。

21. `x_normed` 是不是 GM 上的空间，赋值是不是数据搬运——是。`x_normed_chunk` 在 UB 中计算，转成 BF16 后通过 `TSTORE` 写回 GM。

22. 最后的 `x_normed` 为什么在 `for` 循环里写回，是否应该放到循环外——放在循环里可以按 `[8,128]` 边算边写，不需要在 UB 中保存完整 `[8,4096]` 输出。完整 BF16 输出需要 64 KiB UB；移到循环外是否更好，待验证。

23. 总搬运已经超过 512B，为什么还有提示——PH001 看的是最内层连续宽度，不是总量。当前一次 `[8,128]` BF16 store 总量是 2048B，但实际是 8 行、每行 256B，所以仍低于 a2a3 推荐的 512B。这个只是静态优化提示，不是报错。

24. V-V 操作会不会全部融合成一条指令——不会。当前仍然是 `TROWEXPANDMUL`、`TCOLEXPANDMUL`、`TCVT`、`TSTORE` 等多条指令，只是位于同一个 AIV kernel 中，编译器会做 UB 地址复用和指令调度。

25. `return rms_tid` 返回的是什么——返回整个 RMSNorm SPMD 任务的 TaskId，不是某一个 block 的 ID。调用方可以让后续任务依赖它。

26. 怎么看最终生成了什么 PTOAS 代码——主要看 `ptoas/rms_norm.pto` 和 `ptoas/rms_norm.cpp`；任务提交看 `orchestration/rms_norm_test.cpp`；静态性能提示看 `report/perf_hints.log`。不同写法可能生成完全不同的底层代码，不能只看 Python 源码。

27. 累加、reshape 等底层方案不同，性能怎么保证，PTO ISA 是否有 AscendC 95% 的要求——没有查到“每个 PTO 算子都保证达到 AscendC 95%”的统一规定。文档里提到的是 AscendC 可以用于榨取最后 5%～10% 性能，这是选型建议，不是统一保证。具体算子是否达标仍要固定环境和基线后实测。

## 补充理解

- 数据切分可以简单理解成“先沿 `T` 维分工，再在每个 block 内沿 `D` 维循环”。`T_TILE` 影响逻辑 block 数，`D_TILE` 影响单次计算和搬运大小，两者调的是不同层面。

- `stage=2` 和 `allow_early_resolve=True` 不是一回事。前者优化一个 kernel 内部的 load/compute/store 流水，后者优化不同任务之间的下发时机。

- 两遍 cast、循环内 store 的主要考虑都是控制 UB。当前实现选择“小块读入、小块计算、小块写回”，代价是 `x` 要读取和转换两遍。

- “生成 IR 中使用相同地址”只能说明编译器做了内存复用，不能直接说明前端是原地计算，也不能直接推出存在 bank conflict。

- `reshape`、broadcast mul、rsqrt 等高层写法是否便宜，不能只看 Python 表面。当前文件可以在 `.pto` 中确认 reshape 没有搬运、expand-mul 使用专用指令、高精度 rsqrt 使用 scratch；换一种 shape 或写法后需要重新看生成代码。

- `perf_hints.log` 是静态提示，用来指出值得分析的位置。它不是性能实测，也不是要求必须修改代码。

- 本次明确标为“待验证”的问题包括：tile 是否最优、cast 外提是否会爆 UB、`norm_w` 外提是否更快、同输入 TMUL 是否有 bank conflict，以及 store 方式调整后的真实性能。
