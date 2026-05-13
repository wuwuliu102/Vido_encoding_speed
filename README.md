[jetson_dcvc_rt_endpoint_diagnosis_test_plan.md](https://github.com/user-attachments/files/27712914/jetson_dcvc_rt_endpoint_diagnosis_test_plan.md)
# Vido_encoding_speed# DCVC-RT Jetson 端侧瓶颈诊断与蒸馏提速测试计划表

**文档类型**：公司级测试任务清单 / Execution Checklist  
**项目**：DCVC-RT 码率控制与端侧部署优化  
**平台**：Jetson Orin NX 8G  
**当前阶段**：端侧问题发现、热路径定位、安全替换筛选  
**重要约束**：本阶段先不上服务器训练；所有诊断、profiling、Nsight、TRT、初始化替换安全性测试优先在 Jetson 上完成。  
**最终目的**：找清楚 DCVC-RT 为什么在 Jetson 上慢、慢在哪里、哪些瓶颈适合蒸馏替换、哪些瓶颈只能工程优化、哪些结构说明当前框架不适合端侧。

---

## 0. 总原则

### 0.1 当前阶段不是训练阶段

当前不以“马上训练 student”为目标，而是以**端侧系统诊断**为目标。

必须先回答：

1. Jetson 上 DCVC-RT 的真实瓶颈在哪里？
2. 是 GPU compute 慢，还是 memory-bound，还是 CPU / entropy / Python / IO 慢？
3. 哪些模块是真热路径？
4. 哪些热模块可以被局部替换并通过蒸馏恢复？
5. 哪些热模块初始化一替换就崩，不能进入长训？
6. 哪些瓶颈不是蒸馏能解决的，必须走 TensorRT、CUDA graph、C++、pipeline overlap、IO 优化？
7. 当前 DCVC-RT 框架和 Jetson 硬件之间的不适配点是什么？

### 0.2 Nsight 和 profiling 的定位

之前做过的 Nsight、profiling、TRT 实验**不是废了**，而是后续所有判断的证据基础。

它们的作用是：

- 证明 Jetson 上的真实热路径；
- 证明瓶颈是否来自 GPU compute、memory、kernel launch、CPU gap、entropy、IO；
- 判断 TensorRT 对某个模块是否有用；
- 判断哪些模块值得做局部蒸馏；
- 判断哪些模块不适合蒸馏，只适合工程优化；
- 解释为什么当前非对称 student 初始化质量崩塌。

### 0.3 不允许跳过的门槛

任何模块进入蒸馏前，必须先经过：

```text
Jetson latency profiling
→ Nsight / TRT / 算子原因诊断
→ 单模块替换初始化安全测试
→ bpp / PSNR / x_hat L1 / 闭环稳定性检查
→ 只有安全模块才允许短蒸馏
```

禁止：

```text
看到某个模块耗时高
→ 直接替两个 tail
→ 初始化崩
→ 继续长训等它自己恢复
```

---

## 1. 测试总输出物

最终必须形成以下文件和报告。

### 1.1 环境与基线

- [ ] `jetson_env_report.json`
- [ ] `baseline_teacher_summary.json`
- [ ] `baseline_student_or_candidate_summary.json`，如有候选模型
- [ ] `run_config.yaml`

### 1.2 Full-chain 拆解

- [ ] `full_chain_breakdown.csv`
- [ ] `full_chain_breakdown_summary.json`
- [ ] `full_chain_mode_ablation.md`

### 1.3 模块级 profiling

- [ ] `hot_path_iframe.csv`
- [ ] `hot_path_pframe.csv`
- [ ] `module_latency_rank.md`
- [ ] `module_shape_memory_table.csv`

### 1.4 Nsight Systems

- [ ] `nsys_teacher_q21_1080p.nsys-rep`
- [ ] `nsys_teacher_q21_1080p.sqlite`，如导出
- [ ] `nsys_summary.md`
- [ ] `cpu_gpu_gap_table.csv`
- [ ] `kernel_launch_count_table.csv`

### 1.5 Nsight Compute

- [ ] `ncu_top_kernel.csv`
- [ ] `ncu_module_diagnosis.csv`
- [ ] `ncu_summary.md`

### 1.6 TensorRT 逐模块验证

- [ ] `trt_module_matrix.csv`
- [ ] `trt_layerwise_topk/`
- [ ] `trt_engine_memory_summary.csv`
- [ ] `trt_failure_log.md`

### 1.7 内存、功耗、热稳定性

- [ ] `tegrastats_teacher.log`
- [ ] `tegrastats_candidate.log`
- [ ] `memory_power_summary.csv`
- [ ] `oom_kill_investigation.md`

### 1.8 单模块替换安全性

- [ ] `safe_replacement_matrix.csv`
- [ ] `candidate_init_summary.md`
- [ ] `candidate_visuals/`
- [ ] `candidate_first30_closed_loop.csv`

### 1.9 蒸馏候选决策

- [ ] `distillation_candidate_score.csv`
- [ ] `direction_decision_report.md`
- [ ] `jetson_endpoint_diagnosis_report.md`

---

## 2. 统一测试规范

### 2.1 固定硬件环境

每次正式测试前必须记录：

- [ ] Jetson 型号：Orin NX 8G
- [ ] JetPack 版本
- [ ] CUDA 版本
- [ ] cuDNN 版本
- [ ] TensorRT 版本
- [ ] Python 版本
- [ ] PyTorch 版本
- [ ] ONNX / ONNXRuntime 版本，如使用
- [ ] 当前 `nvpmodel` 模式
- [ ] 是否执行 `jetson_clocks`
- [ ] 初始温度
- [ ] 结束温度
- [ ] 初始 RAM / SWAP 占用
- [ ] 峰值 RAM / SWAP 占用
- [ ] GPU 频率、EMC 频率
- [ ] GR3D 利用率
- [ ] 后台进程情况
- [ ] 是否开启图形桌面
- [ ] 是否有其他 Python / Jupyter / VSCode 进程占用显存或内存

### 2.2 固定模型与输入

至少固定以下配置：

- [ ] I-frame checkpoint：`checkpoints/cvpr2025_image.pth.tar`
- [ ] P-frame checkpoint：`checkpoints/cvpr2025_video.pth.tar`
- [ ] 主测试序列：Kimono1 或当前统一使用的 HEVC_B 1080p 序列
- [ ] 分辨率：至少覆盖 1080p，建议补 720p
- [ ] QP：至少 QP21，建议补 QP25
- [ ] 最大帧数：建议 30 帧用于 Nsight，120 帧用于稳定性
- [ ] warmup 帧数：至少 5 帧
- [ ] 统计时 skip 前 N 帧：建议 skip 10
- [ ] 是否写 bitstream：需要做 on/off ablation
- [ ] 是否计算 PSNR：需要做 on/off ablation
- [ ] 是否保存重建图：仅在视觉检查时打开

### 2.3 统一时间口径

所有结果必须明确区分：

```text
model_forward_ms      模型前向本身
compress_ms           compress 调用整体
src_aligned_ms         对齐输入和测时口径后的压缩时间
full_chain_ms          读帧、转换、padding、模型、熵编码、metric、写文件等完整链路
```

不能用 `full_chain_ms` 直接宣称模型慢，也不能用局部 suffix gain 宣称系统整体快。

### 2.4 统一统计指标

每个时间指标至少输出：

- [ ] mean
- [ ] median / p50
- [ ] p90
- [ ] p95
- [ ] max
- [ ] std
- [ ] coefficient of variation，CV
- [ ] valid frame count
- [ ] warmup / skipped frame count

---

## 3. 阶段 A：环境基线与可复现实验锁定

### 3.1 目标

建立可复现的 Jetson 基线，防止后面所有 profiling 被功耗、温度、内存、后台进程污染。

### 3.2 必测项

- [ ] 记录硬件、软件、驱动版本；
- [ ] 记录当前功耗模式；
- [ ] 测试前执行或明确不执行 `jetson_clocks`；
- [ ] 记录测试前后温度；
- [ ] 记录测试前后内存；
- [ ] 记录测试期间 `tegrastats`；
- [ ] 记录是否出现 swap；
- [ ] 记录是否出现系统 kill；
- [ ] 记录是否出现 thermal throttling；
- [ ] 记录是否有 OOM killer 日志；
- [ ] 保存所有命令行参数。

### 3.3 输出

- [ ] `jetson_env_report.json`
- [ ] `run_config.yaml`
- [ ] `tegrastats_env_baseline.log`

### 3.4 判定

如果测试过程中出现以下情况，结果不得作为正式结论：

- [ ] 温度明显导致降频；
- [ ] RAM 接近极限并进入大量 swap；
- [ ] 后台存在明显抢占 GPU / CPU 的进程；
- [ ] 多次运行结果 CV 异常高且无法解释；
- [ ] 出现 OOM kill 但未记录内存轨迹。

---

## 4. 阶段 B：Full-chain 延迟拆解

### 4.1 目标

解释完整评测链路中每一部分开销，尤其要解释：

```text
compress / src_aligned 为什么是一个数
full_chain 为什么又多出一大截
```

### 4.2 需要拆分的时间项

每帧必须记录：

- [ ] `read_yuv_ms`
- [ ] `colorspace_convert_ms`
- [ ] `h2d_ms`
- [ ] `padding_ms`
- [ ] `model_forward_ms`
- [ ] `entropy_encode_ms`
- [ ] `compress_total_ms`
- [ ] `d2h_ms`
- [ ] `metric_psnr_ms`
- [ ] `write_bitstream_ms`
- [ ] `write_image_or_yuv_ms`
- [ ] `json_log_ms`
- [ ] `sync_overhead_ms`
- [ ] `full_chain_ms`

如不能精确拆分，也必须说明无法拆分的原因和当前计时边界。

### 4.3 必跑 ablation 模式

同一序列、同一 QP、同一帧数下必须跑：

- [ ] `full_normal`：完整流程；
- [ ] `no_metric`：关闭 PSNR / MS-SSIM / 图像质量计算；
- [ ] `no_write_stream`：关闭 bitstream 写入；
- [ ] `no_recon_save`：关闭重建图 / YUV 保存；
- [ ] `synthetic_input`：不用 YUV 读取，直接构造 tensor；
- [ ] `model_only`：只测模型核心 forward / compress；
- [ ] `entropy_off_or_mock`：如果代码允许，用 mock 方式估算 entropy 路径影响；
- [ ] `log_minimal`：关闭详细 JSON 和 verbose log。

### 4.4 输出

- [ ] `full_chain_breakdown.csv`
- [ ] `full_chain_breakdown_summary.json`
- [ ] `full_chain_mode_ablation.md`

### 4.5 结论模板

每个测试结束必须回答：

```text
full_chain 中模型占比是多少？
非模型开销占比是多少？
metric 占比是多少？
IO 占比是多少？
bitstream 写入占比是多少？
Python/logging 占比是多少？
关闭某项后是否显著下降？
```

### 4.6 决策

- 如果 `model_only` 仍然很慢：进入模块级 profiling 和 Nsight。
- 如果 `full_chain` 很慢但 `model_only` 不慢：优先系统工程优化。
- 如果 `metric / IO / logging` 占比很高：这部分不能算作模型蒸馏收益。

---

## 5. 阶段 C：I-frame 模块级 profiling

### 5.1 目标

找出 I-frame 模型的真实热路径，尤其验证已做过的 `dec12` 蒸馏是否是合理候选，以及 `dec11 / dec10 / decoder tail / prior` 是否更值得测。

### 5.2 模块拆分建议

至少包括：

- [ ] image encoder / analysis transform；
- [ ] hyper encoder；
- [ ] hyper decoder；
- [ ] prior fusion；
- [ ] spatial prior；
- [ ] entropy parameter network；
- [ ] decoder / synthesis transform；
- [ ] decoder tail block；
- [ ] `dec10`；
- [ ] `dec11`；
- [ ] `dec12`；
- [ ] postprocess / crop / clamp；
- [ ] entropy coding 相关 CPU / GPU 段，如能拆。

### 5.3 每个模块记录

- [ ] `module_name`
- [ ] `path_type = I-frame`
- [ ] input shape
- [ ] output shape
- [ ] dtype
- [ ] parameters
- [ ] MACs / FLOPs，如可计算
- [ ] mean latency
- [ ] p50 latency
- [ ] p95 latency
- [ ] std
- [ ] 占 `model_forward_ms` 比例
- [ ] peak memory delta
- [ ] 是否含 depthwise conv
- [ ] 是否含 1x1 conv
- [ ] 是否含 pixel shuffle / unshuffle
- [ ] 是否含 concat / split / reshape
- [ ] 是否含 sigmoid / mul / WSiLU
- [ ] 是否可能 TRT-friendly

### 5.4 输出

- [ ] `hot_path_iframe.csv`
- [ ] `hot_path_iframe_summary.md`
- [ ] `iframe_module_shape_memory_table.csv`

### 5.5 必须回答

- [ ] I-frame 最慢 top 10 模块是什么？
- [ ] `dec12` 是否真的是热模块？
- [ ] `dec12` 的速度收益是否足够支撑继续扩展？
- [ ] `dec11 / dec10 / tail block` 是否比 `dec12` 更值得蒸馏？
- [ ] I-frame 优化对长视频平均吞吐贡献是否足够？

---

## 6. 阶段 D：P-frame 模块级 profiling

### 6.1 目标

P-frame 是平均吞吐主战场。必须找出 P-frame 真正热路径，为后续局部蒸馏或 P-frame fast path 提供证据。

### 6.2 模块拆分建议

至少包括：

- [ ] motion estimation / optical flow 相关模块；
- [ ] motion compensation；
- [ ] context extraction；
- [ ] feature extractor；
- [ ] temporal context fusion；
- [ ] encoder / residual analysis；
- [ ] hyper encoder；
- [ ] hyper decoder；
- [ ] prior fusion；
- [ ] spatial prior；
- [ ] entropy parameter network；
- [ ] decoder / synthesis；
- [ ] recon generation；
- [ ] decoder tail；
- [ ] context update；
- [ ] entropy coding；
- [ ] padding / crop / postprocess。

### 6.3 每个模块记录

- [ ] `module_name`
- [ ] `path_type = P-frame`
- [ ] input shape
- [ ] output shape
- [ ] dtype
- [ ] parameters
- [ ] MACs / FLOPs，如可计算
- [ ] mean latency
- [ ] p50 latency
- [ ] p95 latency
- [ ] std
- [ ] 占 P-frame compress 比例
- [ ] 占 full-chain 比例
- [ ] memory delta
- [ ] 算子类型
- [ ] 是否影响 temporal context
- [ ] 是否影响 bpp / prior
- [ ] 是否可能安全替换

### 6.4 输出

- [ ] `hot_path_pframe.csv`
- [ ] `hot_path_pframe_summary.md`
- [ ] `pframe_module_shape_memory_table.csv`

### 6.5 必须回答

- [ ] P-frame top 10 热模块是什么？
- [ ] `feature_extractor` 是否成为新瓶颈？
- [ ] `recon_generation` 是否值得优先做？
- [ ] `decoder` 是否是主瓶颈？
- [ ] `prior / entropy parameter` 是否耗时高但 bpp 风险大？
- [ ] P-frame 哪些模块适合局部蒸馏，哪些不适合？

---

## 7. 阶段 E：Nsight Systems 系统级时间线分析

### 7.1 目标

从系统时间线角度解释 Jetson 为什么慢，区分：

```text
GPU compute-bound
CPU-bound
memory-bound
launch-bound
sync-bound
IO-bound
entropy-bound
```

### 7.2 采集要求

建议只跑 30 帧，避免文件过大。

必须覆盖：

- [ ] teacher baseline；
- [ ] 当前 student / candidate，如能跑；
- [ ] no metric 模式；
- [ ] synthetic input 模式；
- [ ] 至少一组 1080p QP21。

### 7.3 Nsight Systems 必查项

- [ ] GPU 是否连续忙；
- [ ] CPU 是否等待 GPU；
- [ ] GPU 是否等待 CPU；
- [ ] kernel launch 是否碎片化；
- [ ] 每帧 kernel launch 数；
- [ ] 大量小 kernel 是否存在；
- [ ] `cudaDeviceSynchronize` / `torch.cuda.synchronize` 出现位置；
- [ ] H2D / D2H 是否阻塞；
- [ ] entropy coding 是否与 GPU 网络串行；
- [ ] Python loop 间隙是否明显；
- [ ] 是否存在长时间 CPU 空洞；
- [ ] 是否存在显著 GPU idle gap；
- [ ] 是否存在 TensorRT engine 与 PyTorch 之间切换开销；
- [ ] 是否存在 IO / metric 导致 GPU 等待。

### 7.4 输出

- [ ] `nsys_teacher_q21_1080p.nsys-rep`
- [ ] `nsys_summary.md`
- [ ] `cpu_gpu_gap_table.csv`
- [ ] `kernel_launch_count_table.csv`
- [ ] `sync_callsite_table.csv`

### 7.5 必须回答

- [ ] DCVC-RT 在 Jetson 上主要是 GPU 忙，还是 CPU/GPU 互相等待？
- [ ] kernel launch 是否过多？
- [ ] 是否存在明显同步点？
- [ ] entropy coding 是否可以 overlap？
- [ ] 是否需要 pipeline parallel / parallel entropy coding？
- [ ] 是否需要减少 Python 包装？
- [ ] 是否需要 C++/CUDA extension？

---

## 8. 阶段 F：Nsight Compute 热 kernel 分析

### 8.1 目标

对 top 热模块和 top 热 kernel 做微观诊断，判断它们为什么慢：

- compute-bound；
- memory-bound；
- Tensor Core 利用率低；
- occupancy 低；
- depthwise / shuffle 不友好；
- kernel launch 多；
- layout transform 多。

### 8.2 采样范围

从模块级 profiling 和 Nsight Systems 中选 top 5～10 个热 kernel / 热模块，不要全模型乱采。

候选包括：

- [ ] decoder 相关 conv；
- [ ] recon generation；
- [ ] feature extractor；
- [ ] spatial prior；
- [ ] prior fusion；
- [ ] hyper decoder；
- [ ] pixel shuffle / depth-to-space；
- [ ] depthwise conv；
- [ ] 1x1 conv；
- [ ] WSiLU / sigmoid / mul 相关 kernel。

### 8.3 必查指标

- [ ] kernel duration；
- [ ] achieved occupancy；
- [ ] SM utilization；
- [ ] Tensor Core utilization；
- [ ] FP16 utilization；
- [ ] DRAM throughput；
- [ ] L2 hit rate；
- [ ] memory load efficiency；
- [ ] memory store efficiency；
- [ ] warp stall reason；
- [ ] register usage；
- [ ] shared memory usage；
- [ ] launch overhead；
- [ ] 是否存在非 coalesced memory access；
- [ ] 是否存在大量 layout transform。

### 8.4 输出

- [ ] `ncu_top_kernel.csv`
- [ ] `ncu_module_diagnosis.csv`
- [ ] `ncu_summary.md`

### 8.5 诊断规则

如果出现：

```text
Tensor Core utilization 低 + DRAM throughput 高 + occupancy 低
```

判断为 memory-bound / 小算子不友好。

如果出现：

```text
kernel 很多但每个很短
```

判断为 launch-bound / 模块碎片化。

如果出现：

```text
conv kernel 很长且 Tensor Core 利用率高
```

判断为真实 compute-heavy，可考虑降 channel / 降 block / 蒸馏。

如果出现：

```text
shuffle / pixelshuffle / reshape 很重
```

判断为 layout transform 问题，需要结构减少重排或融合。

---

## 9. 阶段 G：TensorRT 逐模块验证

### 9.1 目标

TensorRT 不是主方法，但必须作为端侧部署证据，用来判断：

- 某模块是否 TRT-friendly；
- TRT 后是否仍然慢；
- 是否有 layer fallback；
- 是否被 shuffle / layout transform 拖慢；
- engine memory 是否过大；
- PyTorch + TRT 混合是否引入额外同步和内存压力。

### 9.2 测试对象

优先对以下模块做单独 ONNX / TRT：

- [ ] I-frame decoder；
- [ ] I-frame dec12 / dec11 / dec10；
- [ ] P-frame feature extractor；
- [ ] P-frame decoder；
- [ ] P-frame recon generation；
- [ ] P-frame hyper decoder；
- [ ] P-frame prior fusion；
- [ ] P-frame spatial prior；
- [ ] 当前 student / candidate suffix；
- [ ] teacher suffix 对照。

### 9.3 trtexec 记录项

每个 engine 必须记录：

- [ ] build 是否成功；
- [ ] ONNX 导出是否成功；
- [ ] TensorRT 版本；
- [ ] FP16 是否开启；
- [ ] 是否有 FP32 fallback；
- [ ] 是否有 unsupported op；
- [ ] engine size；
- [ ] activation memory；
- [ ] persistent memory；
- [ ] scratch memory；
- [ ] H2D latency；
- [ ] D2H latency；
- [ ] GPU compute mean；
- [ ] GPU compute p50；
- [ ] GPU compute p95；
- [ ] GPU compute CV；
- [ ] throughput；
- [ ] layer-wise top 20 latency；
- [ ] 是否存在大量 `SHUFFLE`；
- [ ] 是否存在 `DepthToSpace`；
- [ ] 是否有 FP16 subnormal warning；
- [ ] 是否有 cuDNN / cuBLAS 初始化异常；
- [ ] 是否有 tactic 选择异常。

### 9.4 输出

- [ ] `trt_module_matrix.csv`
- [ ] `trt_layerwise_topk/<module_name>.csv`
- [ ] `trt_engine_memory_summary.csv`
- [ ] `trt_failure_log.md`

### 9.5 结论分类

每个模块必须分类：

```text
A 类：TRT 明显加速，适合作为部署后端；
B 类：TRT 小幅加速，结构仍需蒸馏 / 重设；
C 类：TRT 后仍然很慢，说明结构与 Jetson 不匹配；
D 类：TRT 不稳定或内存压力大，不适合整链路；
E 类：ONNX / TRT 不支持，需要替换算子或保留 PyTorch。
```

---

## 10. 阶段 H：内存、功耗、热稳定性测试

### 10.1 目标

解释系统被 kill、TRT 后不稳定、full-chain 波动大的原因。

### 10.2 必测场景

- [ ] teacher PyTorch；
- [ ] teacher TRT partial；
- [ ] student / candidate PyTorch；
- [ ] student / candidate TRT partial；
- [ ] teacher + student 同时加载；
- [ ] teacher + student + TRT engine 同时加载；
- [ ] no metric；
- [ ] synthetic input；
- [ ] 30 帧短跑；
- [ ] 120 帧长跑。

### 10.3 tegrastats 必记字段

- [ ] RAM；
- [ ] SWAP；
- [ ] CPU 使用率；
- [ ] GPU / GR3D 使用率；
- [ ] EMC；
- [ ] temperature；
- [ ] power；
- [ ] frequency；
- [ ] 是否进入 swap；
- [ ] 是否 OOM kill；
- [ ] kill 前内存走势；
- [ ] 是否 thermal throttling。

### 10.4 输出

- [ ] `tegrastats_teacher.log`
- [ ] `tegrastats_candidate.log`
- [ ] `memory_power_summary.csv`
- [ ] `oom_kill_investigation.md`

### 10.5 决策

如果 RAM 接近极限：

- [ ] 不允许同时加载 teacher / student / 多个 TRT engine；
- [ ] 蒸馏训练不在 Jetson 做；
- [ ] 测试脚本必须分阶段释放模型；
- [ ] 减少中间 tensor 保存；
- [ ] 固定 shape；
- [ ] 预分配 buffer；
- [ ] 避免 PyTorch / TensorRT 来回切换；
- [ ] 检查是否需要关闭重建图保存、metric、verbose JSON。

---

## 11. 阶段 I：单模块替换初始化安全测试

### 11.1 目标

解决当前最大问题：**热模块不等于可替换模块**。

每个候选模块必须先单独替换，不训练，直接测初始化质量。

### 11.2 候选模块

I-frame：

- [ ] `dec12`
- [ ] `dec11`
- [ ] `dec10`
- [ ] single decoder tail
- [ ] hyper decoder block
- [ ] prior fusion block
- [ ] spatial prior block

P-frame：

- [ ] feature extractor
- [ ] temporal context fusion
- [ ] recon generation
- [ ] decoder block
- [ ] decoder tail
- [ ] hyper decoder
- [ ] prior fusion
- [ ] spatial prior
- [ ] entropy parameter network
- [ ] context update

### 11.3 替换原则

一次只替换一个模块。

禁止：

```text
一次替两个 tail
一次替整条 decoder
一次替 feature extractor + recon
一次替 prior + decoder
```

只有当两个模块分别单独通过初始化安全测试并完成短蒸馏后，才允许 joint finetune。

### 11.4 每个候选记录

- [ ] candidate name；
- [ ] replaced module；
- [ ] teacher latency；
- [ ] candidate latency；
- [ ] latency gain；
- [ ] init PSNR；
- [ ] teacher PSNR；
- [ ] init PSNR drop；
- [ ] init bpp；
- [ ] teacher bpp；
- [ ] init bpp change；
- [ ] `x_hat` L1 to teacher；
- [ ] feature L1 to teacher，如可取；
- [ ] prior parameter L1，如涉及 prior；
- [ ] first 30 frames PSNR curve；
- [ ] first 30 frames bpp curve；
- [ ] 是否出现累积崩塌；
- [ ] 是否偏色；
- [ ] 是否块效应；
- [ ] 是否明显糊；
- [ ] 是否 temporal flicker；
- [ ] 是否系统稳定；
- [ ] 内存变化；
- [ ] 是否值得短蒸馏。

### 11.5 判定规则

```text
PSNR drop < 0.3 dB
且 bpp 不明显上涨
且图像不崩
且前 30 帧不累积恶化
→ A 类，优先蒸馏
```

```text
PSNR drop 0.3 ~ 1.0 dB
或 bpp 小幅上涨
但图像未崩
→ B 类，允许短蒸馏验证
```

```text
PSNR drop > 1.0 dB
或图像明显崩
或 bpp 明显爆
或 P-frame 闭环累积崩
→ C 类，当前替换结构不合格，不允许长训
```

```text
latency gain 很小
即使质量安全
→ D 类，只作为消融，不作为主线
```

### 11.6 输出

- [ ] `safe_replacement_matrix.csv`
- [ ] `candidate_init_summary.md`
- [ ] `candidate_visuals/`
- [ ] `candidate_first30_closed_loop.csv`

---

## 12. 阶段 J：蒸馏候选打分

### 12.1 目标

用统一标准决定哪些模块值得蒸馏，而不是靠直觉。

### 12.2 候选评分公式

建议记录：

```text
DistillScore =
LatencyShare
× ReplaceGain
× InitSafety
× TRTFriendliness
÷ BppRisk
```

也可以使用表格形式，不一定真的做数学归一化，但每一项必须有证据。

### 12.3 评分字段

- [ ] `module_name`
- [ ] `path_type`
- [ ] `latency_share`
- [ ] `absolute_gain_ms`
- [ ] `init_safety_level`
- [ ] `bpp_risk_level`
- [ ] `temporal_risk_level`
- [ ] `trt_friendliness`
- [ ] `memory_risk`
- [ ] `distill_difficulty`
- [ ] `expected_paper_value`
- [ ] `final_priority`

### 12.4 输出

- [ ] `distillation_candidate_score.csv`
- [ ] `distillation_candidate_rank.md`

### 12.5 优先级解释

优先进入蒸馏的模块应满足：

```text
Jetson 上热
替换后有明显 latency gain
初始化不崩
bpp 风险可控
闭环稳定
蒸馏目标清楚
```

---

## 13. 阶段 K：短蒸馏验证，不做长训

### 13.1 目标

对通过安全筛选的模块进行短蒸馏，验证 loss 和质量是否能回收。

### 13.2 本阶段仍以 Jetson 诊断为主

本阶段不要求在 Jetson 上训练。  
如果 Jetson 资源不足，可以只在 Jetson 上做：

- [ ] 初始化测试；
- [ ] 小 batch 推理；
- [ ] latency 验证；
- [ ] closed-loop 验证。

训练本身后续可放到服务器，但当前计划的核心是把 Jetson 端侧问题找全。

### 13.3 短蒸馏门槛

每个候选只做短程验证：

- [ ] 1k step：loss 是否下降；
- [ ] 5k step：PSNR 是否明显回收；
- [ ] 20k step：是否接近 teacher；
- [ ] 短训后 bpp 是否可控；
- [ ] P-frame 是否闭环稳定。

### 13.4 不能长训的情况

- [ ] 初始化已经崩；
- [ ] 5k / 20k 无明显回收；
- [ ] PSNR 回来但 bpp 爆；
- [ ] 单帧正常但 P-frame 累积崩；
- [ ] latency gain 太小；
- [ ] 内存压力不可接受；
- [ ] TRT 或部署路径不可行。

---

## 14. 阶段 L：不能蒸馏的瓶颈处理方向

### 14.1 Entropy coding / CPU gap

如果 Nsight 显示瓶颈在 entropy coding 或 CPU/GPU gap：

可选方向：

- [ ] entropy coding 与网络推理 overlap；
- [ ] parallel coding；
- [ ] C++ bitstream path；
- [ ] 减少 Python 参与；
- [ ] 减少同步点；
- [ ] 固定 bitstream path；
- [ ] 异步 pipeline。

不应把这类瓶颈归因于 student 不够小。

### 14.2 IO / metric / logging

如果 full-chain 多出来的开销主要来自：

- [ ] YUV 读取；
- [ ] YUV420 / RGB 转换；
- [ ] PSNR / MS-SSIM；
- [ ] JSON log；
- [ ] 重建图保存；
- [ ] bitstream 写入；

则方向是：

- [ ] 关闭或延迟 metric；
- [ ] 预读帧；
- [ ] buffer reuse；
- [ ] C++ / CUDA pre/post；
- [ ] minimal logging；
- [ ] benchmark 与真实系统路径分开。

### 14.3 TensorRT engine 拼接开销

如果 PyTorch + TRT 混合导致同步、转换、内存压力：

- [ ] 避免多个小 engine 频繁切换；
- [ ] 只保留最有效 engine；
- [ ] 统一 FP16 IO；
- [ ] 固定 shape；
- [ ] engine 常驻；
- [ ] 避免 teacher/student/TRT 同时加载；
- [ ] 检查 PyTorch tensor 与 TRT buffer 转换成本。

### 14.4 Memory / OOM

如果主要问题是内存：

- [ ] 降低 activation size；
- [ ] 预分配 buffer；
- [ ] 释放不需要的 teacher；
- [ ] 分阶段加载模型；
- [ ] 关闭中间结果保存；
- [ ] 关闭 verbose；
- [ ] 不在 Jetson 上训练；
- [ ] 避免多模型常驻；
- [ ] 检查 dataloader / reader 缓存。

---

## 15. 最终方向决策树

### 15.1 进入局部蒸馏

条件：

- [ ] 模块是热路径；
- [ ] 替换后 latency gain 明显；
- [ ] 初始化 PSNR drop 小；
- [ ] bpp 不爆；
- [ ] 前 30 帧闭环稳定；
- [ ] 蒸馏目标明确；
- [ ] 部署路径可行。

方向：

```text
single-module local distillation
→ x_hat / feature / ctx / prior loss
→ closed-loop RD
→ Jetson latency validation
```

### 15.2 结构重设但不长训当前版本

条件：

- [ ] 模块很热；
- [ ] 当前替换初始化崩；
- [ ] 但理论收益很大。

方向：

- [ ] 缩小替换范围；
- [ ] 加 adapter；
- [ ] residual student block；
- [ ] Net2Net / weight inheritance；
- [ ] 保持输入输出 channel；
- [ ] 分层逐个替换；
- [ ] 不允许两个 tail 同时随机替换。

### 15.3 工程优化

条件：

- [ ] Nsight 显示 CPU gap / entropy / IO / logging / sync 是主要问题；
- [ ] 蒸馏无法解决。

方向：

- [ ] pipeline overlap；
- [ ] CUDA graph；
- [ ] static shape；
- [ ] buffer pre-allocation；
- [ ] C++ extension；
- [ ] minimal IO / logging；
- [ ] TRT-only selected modules。

### 15.4 放弃候选

条件：

- [ ] latency gain 小；
- [ ] 初始化崩；
- [ ] bpp 爆；
- [ ] P-frame 累积崩；
- [ ] TRT 不可部署；
- [ ] 内存风险高；
- [ ] 蒸馏短训无恢复。

---

## 16. 重点候选模块优先级

### 16.1 第一优先级：P-frame 热路径

P-frame 是平均吞吐主战场。

优先看：

- [ ] feature extractor；
- [ ] recon generation；
- [ ] decoder；
- [ ] decoder tail；
- [ ] temporal context fusion；
- [ ] context update。

理由：

```text
P-frame 数量远多于 I-frame；
P-frame 优化对平均速度贡献更大；
但必须检查 temporal 累积误差。
```

### 16.2 第二优先级：I-frame decoder tail

包括：

- [ ] `dec12`
- [ ] `dec11`
- [ ] `dec10`
- [ ] single tail block

理由：

```text
I-frame dec12 已经做过蒸馏；
适合作为方法 pilot；
但长视频平均收益可能不够，不能只停留在 I-frame。
```

### 16.3 第三优先级：prior / entropy parameter

包括：

- [ ] prior fusion；
- [ ] spatial prior；
- [ ] entropy parameter network；
- [ ] hyper decoder。

理由：

```text
可能耗时高；
但 bpp 风险大；
必须加 prior / latent / bits loss；
不能只看 PSNR。
```

### 16.4 第四优先级：IO / entropy / system path

理由：

```text
这部分可能显著影响 full-chain；
但不是蒸馏解决；
应作为系统优化分支。
```

---

## 17. 当前已知失败经验必须纳入计划

### 17.1 非对称 student 初始化质量崩塌

结论：

```text
当前一次性替换两个 tail 的非对称结构不合理；
不能继续长训；
必须改为单模块替换 + 初始化安全筛选。
```

### 17.2 I-frame dec12 蒸馏

结论：

```text
局部单层蒸馏有价值；
但 dec12 只是 pilot；
需要确认真实 Jetson gain；
不能直接扩成两个 tail。
```

### 17.3 skip / lite / 硬砍模块

结论：

```text
硬跳高耗时模块可以省时间；
但质量和 bpp 容易崩；
说明热模块重要，必须蒸馏或结构重设。
```

### 17.4 DCT / side-bit / patch repair

结论：

```text
外挂修复收益不稳定；
不作为当前主线；
只可作为失败对照和论文动机。
```

### 17.5 TensorRT 实验

结论：

```text
TRT 能帮助判断部署友好性；
但不能一键救 DCVC-RT；
如果 TRT 后 GPU compute 仍高，说明结构本身与 Jetson 不匹配。
```

### 17.6 Rate control

结论：

```text
rate control 能削峰和调码率；
不能解决模型慢；
可作为系统控制辅助，不作为端侧加速主线。
```

---

## 18. 结果记录模板

### 18.1 Full-chain 表模板

| frame | mode | read_yuv | convert | h2d | padding | model | entropy | d2h | metric | write | log | full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 18.2 Module profiling 表模板

| module | path | mean ms | p50 | p95 | std | share | input shape | output shape | dtype | ops | memory delta |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---:|

### 18.3 Nsight Systems 表模板

| item | value | evidence | interpretation | next action |
|---|---:|---|---|---|
| GPU busy ratio |  |  |  |  |
| kernel launches / frame |  |  |  |  |
| largest GPU idle gap |  |  |  |  |
| sync count |  |  |  |  |
| CPU wait time |  |  |  |  |
| entropy overlap potential |  |  |  |  |

### 18.4 TRT 表模板

| module | build | fp16 | GPU mean | p95 | CV | H2D | D2H | engine MB | activation MB | top slow layer | class |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|

### 18.5 安全替换表模板

| candidate | replaced module | latency gain | init PSNR drop | bpp change | x_hat L1 | first30 stable | visual | memory | decision |
|---|---|---:|---:|---:|---:|---|---|---:|---|

### 18.6 蒸馏候选表模板

| module | hot path rank | gain | init safety | bpp risk | temporal risk | TRT friendly | memory risk | score | priority |
|---|---:|---:|---|---|---|---|---|---:|---|

---

## 19. 一周执行计划

### Day 1：环境与 full-chain

- [ ] 完成 `jetson_env_report.json`
- [ ] 跑 teacher baseline
- [ ] 跑 full / no metric / no write / synthetic / model-only
- [ ] 生成 `full_chain_breakdown.csv`
- [ ] 初步判断 full-chain 额外开销来源

### Day 2：I-frame / P-frame module profiling

- [ ] 完成 I-frame 热路径表
- [ ] 完成 P-frame 热路径表
- [ ] 输出 top 10 热模块
- [ ] 判断 I-frame dec12 是否值得继续扩展
- [ ] 判断 P-frame 主战场模块

### Day 3：Nsight Systems

- [ ] 采集 teacher 1080p QP21 30 帧
- [ ] 采集 no metric / synthetic 对照
- [ ] 统计 kernel launch、CPU/GPU gap、sync
- [ ] 输出 `nsys_summary.md`

### Day 4：Nsight Compute + TensorRT 逐模块

- [ ] 对 top 热 kernel 做 NCU
- [ ] 对 top 候选模块做 TRT
- [ ] 输出 `ncu_module_diagnosis.csv`
- [ ] 输出 `trt_module_matrix.csv`

### Day 5：内存、功耗、热稳定性

- [ ] 跑 teacher / candidate tegrastats
- [ ] 复现或解释系统 kill
- [ ] 输出 `memory_power_summary.csv`
- [ ] 输出 `oom_kill_investigation.md`

### Day 6：单模块替换初始化安全测试

- [ ] I-frame dec12 / dec11 / dec10 单独测试
- [ ] P-frame feature / recon / decoder / prior 单独测试
- [ ] 输出 `safe_replacement_matrix.csv`
- [ ] 生成 candidate visuals
- [ ] 标注 A/B/C/D 类候选

### Day 7：总报告与方向决策

- [ ] 汇总所有表
- [ ] 生成 `jetson_endpoint_diagnosis_report.md`
- [ ] 生成 `direction_decision_report.md`
- [ ] 决定：局部蒸馏 / 结构重设 / 工程优化 / 放弃候选

---

## 20. 代码与命令管理要求

### 20.1 每次运行必须保存

- [ ] 命令行；
- [ ] git commit hash；
- [ ] 修改过的文件列表；
- [ ] checkpoint 路径；
- [ ] 输入序列；
- [ ] QP；
- [ ] 分辨率；
- [ ] max frames；
- [ ] warmup；
- [ ] skip；
- [ ] 输出目录；
- [ ] 是否开启 TensorRT；
- [ ] 是否开启 FP16；
- [ ] 是否开启 channels_last；
- [ ] 是否开启 cudnn benchmark；
- [ ] 是否开启 TF32；
- [ ] 是否写 bitstream；
- [ ] 是否计算 metric。

### 20.2 输出目录建议

```text
out/jetson_diagnosis/
  env/
  full_chain/
  module_profile/
  nsys/
  ncu/
  trt/
  tegrastats/
  replacement_safety/
  candidate_visuals/
  reports/
```

### 20.3 命名规范

```text
{date}_{device}_{model}_{seq}_{resolution}_qp{qp}_{mode}
```

示例：

```text
2026xxxx_orinnx8g_teacher_kimono1_1080p_qp21_full
2026xxxx_orinnx8g_teacher_kimono1_1080p_qp21_no_metric
2026xxxx_orinnx8g_candidate_dec12_kimono1_1080p_qp21_init
```

---

## 21. 最终报告必须包含的结论

`jetson_endpoint_diagnosis_report.md` 必须回答：

1. Jetson 上 DCVC-RT 慢的主因是什么？
2. full-chain 中模型和非模型开销分别占多少？
3. P-frame 和 I-frame 的热路径分别是什么？
4. Nsight Systems 显示 GPU/CPU 关系是什么？
5. Nsight Compute 显示热 kernel 是 compute-bound、memory-bound、launch-bound 还是 shuffle-bound？
6. TensorRT 对哪些模块有效，对哪些模块无效？
7. 系统被 kill 是否由内存导致？
8. 哪些模块可以进入局部蒸馏？
9. 哪些模块初始化崩塌，不能长训？
10. 哪些瓶颈必须工程优化？
11. 当前非对称 student 为什么失败？
12. dec12 蒸馏在整体路线中的位置是什么？
13. 下一阶段是否应该：
    - 局部蒸馏；
    - P-frame fast path；
    - 复杂度可伸缩 codec；
    - entropy / pipeline 工程优化；
    - TensorRT/static-shape 部署；
    - 还是框架重构？

---

## 22. 禁止项

以下行为禁止作为正式实验结论：

- [ ] 只看 full-chain 就说模型慢；
- [ ] 只看 suffix gain 就说系统快；
- [ ] 没有 warmup / skip 就报平均时间；
- [ ] 没有记录温度 / 功耗 / 内存就比较 latency；
- [ ] 没有 no metric / no write ablation 就归因模型；
- [ ] 没有 Nsight 就声称 CPU/GPU 瓶颈；
- [ ] 没有单模块初始化测试就进入长蒸馏；
- [ ] 两个 tail 同时替换后崩，还继续长训；
- [ ] 只看 PSNR，不看 bpp；
- [ ] 只看单帧，不看 P-frame 闭环；
- [ ] TensorRT 跑通就认为部署成功；
- [ ] 忽略 OOM / swap / throttling；
- [ ] 重启 side-bit / DCT 作为主线；
- [ ] 把 rate control 当成端侧加速主方法；
- [ ] 在 Jetson 上做不必要的完整训练。

---

## 23. 最终一句话目标

本计划最终要产出一张总表：

| 模块 | Jetson 耗时 | Nsight 原因 | TRT 表现 | 初始化替换质量 | bpp 风险 | 是否可蒸馏 | 下一步 |
|---|---:|---|---|---|---|---|---|

只有这张表出来后，才允许进入下一阶段训练或框架改造。

当前核心不是“蒸馏能不能行”，而是：

```text
用 Jetson 证据找出哪些东西值得蒸馏，
哪些东西不能蒸馏，
哪些东西必须工程优化，
哪些东西说明 DCVC-RT 框架本身不适合 Orin NX。
```
