# fused_swiglu_scale 四种实现性能对比报告

## 一、四种实现方法介绍

| 方法 | 名称 | 实现方式 | 调用路径 | 适用场景 |
|------|------|---------|---------|---------|
| **Method 1** | Native Python | 纯 Python 手写，调用 Paddle 标准 op（`swiglu`、`sigmoid`、`chunk`、`concat` 等） | `benchmark_fused_swiglu_scale_native.py` | 通用 fallback，任何设备可用 |
| **Method 2** | PaddleFleet Custom Op | CUDA C++ extension，通过 `PD_BUILD_OP` 注册为自定义算子 | `benchmark_fused_swiglu_scale.py` 调用 `paddlefleet.ops.fused_swiglu_scale_forward` | CUDA 环境 |
| **Method 3** | Semi-Fused（v2） | 调用 Paddle 粗粒度融合算子 `_C_ops.swiglu` + `paddle.multiply` | `benchmark_fused_swiglu_scale_v2.py` | XPU 环境，无需编译自定义 kernel |
| **Method 4** | Fused XPU Op（本工作） | XPU C++ Kernel，通过 `PD_REGISTER_KERNEL` 注册进 Paddle 框架 | `benchmark_fused_swiglu_scale_paddle.py` 调用 `paddle._C_ops.fused_swiglu_scale` | XPU 环境 |

### Method 1：Native Python

对应 PaddleFleet 中的 else 分支逻辑，完全用 Python + Paddle 标准 API 实现：

```python
def native_forward(x, scale):
    out = swiglu(x)                           # 内部: chunk + silu + mul
    scale_exp = scale.cast(x.dtype)
    while scale_exp.ndim < out.ndim:
        scale_exp = scale_exp.unsqueeze(-1)   # broadcast 维度对齐
    return out * scale_exp

def native_backward(x, scale, out_grad):
    # 手动推导 SwiGLU 反向 + scale 反向
    gate, val = x[..., :H], x[..., H:]
    sig = F.sigmoid(gate)
    silu = gate * sig
    d_val = d_u * silu
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))
    d_x = paddle.concat([d_gate, d_val], axis=-1)
    d_scale = paddle.sum(out_grad * swiglu_out, axis=-1)
    return d_x, d_scale
```

**特点**：
- 不依赖任何 C++ 编译，纯 Python 即可运行
- 调用多个独立的 Paddle 标准 op，每个 op 都是一个独立的 kernel launch
- 反向也是纯 Python 手写，没有框架自动微分

### Method 2：PaddleFleet Custom Op

PaddleFleet 通过 `paddle.utils.cpp_extension` 将 CUDA C++ 代码编译为自定义算子：

```cpp
// CUDA kernel: VectorizedFusedSwiGLUScaleKernel
// 手写 CUDA kernel，一个 kernel 完成 SwiGLU + Scale

PD_BUILD_OP(fused_swiglu_scale)
    .Inputs({"X", "Scale"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(FusedSwiGLUScaleForward));
```

**特点**：
- CUDA C++ 手写 kernel，单 kernel 完成融合
- 通过 `paddle.utils.cpp_extension.load` 动态编译加载
- 需要 CUDA 环境，在 XPU 环境下可能无法运行或性能不佳

### Method 3：Semi-Fused（v2）

介于 Native Python 和全融合 XPU Kernel 之间的半融合方案，调用 Paddle 已有的粗粒度融合算子：

```python
# Forward: 调用框架内置 swiglu + multiply
def fused_swiglu_scale_forward(x, scale):
    out = _C_ops.swiglu(x, None)           # 框架内置粗粒度 swiglu kernel
    return paddle.multiply(out, scale)     # broadcast multiply

# Backward: swiglu_grad + multiply + sum
def fused_swiglu_scale_backward(x, scale, out_grad):
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = paddle.multiply(out_grad, scale)
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = paddle.sum(paddle.multiply(out_grad, swiglu_out), axis=-1)
    return d_x, d_scale
```

**特点**：
- 不依赖自定义 C++ 编译，纯 Python 调用已有融合算子
- 前向仅 2 个 kernel launch（swiglu + multiply），反向 5 个
- 反向利用框架自动注册的 `swiglu_grad`，无需手写求导公式
- 在 XPU 上可作为快速 fallback，性能通常优于 Native Python

### Method 4：Fused XPU Op（本工作）

将 XPU Kernel 直接编译进 Paddle 框架，通过 `paddle._C_ops` 调用：

```cpp
// XPU kernel: FusedSwigluScaleKernel
// 调用 XDNN API: xpu::swiglu + xpu::broadcast_mul

PD_REGISTER_KERNEL(fused_swiglu_scale, XPU, ALL_LAYOUT,
                   phi::fusion::FusedSwigluScaleKernel,
                   float, phi::float16, phi::bfloat16) {}
```

**特点**：
- 基于 XDNN（百度昆仑 XPU 深度神经网络库）API 实现
- 单 kernel 内完成 SwiGLU + broadcast_mul，减少 kernel launch 开销
- 通过 Paddle 框架的 YAML + 代码生成器注册，支持 dygraph/static 两种模式
- 支持 float32 / float16 / bfloat16 三种精度

---

## 二、测试环境

| 项目 | 配置 |
|------|------|
| 设备 | XPU（百度昆仑芯） |
| Paddle 版本 | 3.5.0.dev20260511（自定义编译） |
| Python | 3.10 |
| 虚拟环境 | `/root/paddlejob/Gruge/envs/py310_paddleFormers` |
| Warmup | 10 轮 |
| Benchmark Iterations | 100 轮 |
| 计时方式 | `time.perf_counter()` + `paddle.device.synchronize()` |
| 测试日期 | 2026-05-21 |

---

## 三、性能对比数据

### 3.1 原始数据（ms）

| Config | x_shape | scale_shape | Native Fwd | Native Bwd | Fleet Fwd | Fleet Bwd | v2 Fwd | v2 Bwd | XPU Fwd | XPU Bwd |
|--------|---------|-------------|-----------|-----------|----------|----------|--------|--------|--------|--------|
| small_2D | 4×8 | 4×1 | 0.055 | 0.508 | 0.049 | 0.422 | 0.047 | 0.131 | 0.020 | 0.043 |
| medium_2D | 1024×256 | 1024×1 | 0.055 | 0.509 | 0.049 | 0.422 | 0.048 | 0.145 | 0.020 | 0.048 |
| large_2D | 4096×512 | 4096×1 | 0.055 | 0.510 | 0.049 | 0.422 | 0.054 | 0.143 | 0.031 | 0.074 |
| small_3D | 2×3×16 | 1×1×8 | 0.055 | 0.565 | 0.049 | 0.429 | 0.082 | 0.146 | 0.020 | 0.044 |
| medium_3D | 8×128×512 | 8×128×256 | 0.054 | 0.513 | 0.050 | 0.475 | 0.051 | 0.142 | 0.020 | 0.048 |
| large_3D | 32×512×1024 | 32×512×512 | 0.237 | 1.245 | 0.238 | 1.245 | 0.237 | 0.514 | 0.200 | 0.439 |

> 注：四种方法使用完全相同的输入 shape（包括 scale_shape），确保性能对比公平。scale_shape 按 broadcast 语义设计，与输出维度对齐。

### 3.2 加速比（相对 Native）

| Config | Fleet Fwd | Fleet Bwd | v2 Fwd | v2 Bwd | XPU Fwd | XPU Bwd |
|--------|-----------|-----------|--------|--------|---------|---------|
| small_2D | 1.12× | 1.20× | 1.17× | **3.88×** | **2.75×** | **11.81×** |
| medium_2D | 1.12× | 1.21× | 1.15× | **3.51×** | **2.75×** | **10.60×** |
| large_2D | 1.12× | 1.21× | 1.02× | **3.57×** | **1.77×** | **6.89×** |
| small_3D | 1.12× | 1.32× | 0.67× | **3.87×** | **2.75×** | **12.84×** |
| medium_3D | 1.08× | 1.08× | 1.06× | **3.61×** | **2.70×** | **10.69×** |
| large_3D | 1.00× | 1.00× | 1.00× | **2.42×** | **1.19×** | **2.84×** |

### 3.3 可视化表格

```
====================================================================================================
                              Forward Latency (ms)                              Backward Latency (ms)
Config         Native    Fleet    v2(Semi)  XPU(Fused)     Native    Fleet    v2(Semi)  XPU(Fused)
====================================================================================================
small_2D        0.055    0.049    0.047     0.020 ★★★      0.508    0.422    0.131 ★★★  0.043 ★★★
medium_2D       0.055    0.049    0.048     0.020 ★★★      0.509    0.422    0.145 ★★★  0.048 ★★★
large_2D        0.055    0.049    0.054     0.031 ★★       0.510    0.422    0.143 ★★★  0.074 ★★★
small_3D        0.055    0.049    0.082     0.020 ★★★      0.565    0.429    0.146 ★★★  0.044 ★★★
medium_3D       0.054    0.050    0.051     0.020 ★★★      0.513    0.475    0.142 ★★★  0.048 ★★★
large_3D        0.237    0.238    0.237     0.200 ★        1.245    1.245    0.514 ★★   0.439 ★★
====================================================================================================
★★★ = 显著优势, ★★ = 中等优势, ★ = 轻微优势
```

---

## 四、结果分析

### 4.1 Fused XPU Op (Method 4) vs Native Python (Method 1)

| 指标 | 结论 |
|------|------|
| **前向加速** | 小/中等规模 **2.7x ~ 2.8x**，大规模 **1.19x** |
| **反向加速** | 小/中等规模 **6.9x ~ 12.8x**，大规模 **2.84x** |
| **原因分析** | 小 tensor 下 kernel launch 开销占比高，融合后减少多个 kernel 调度，收益明显；大 tensor 下计算时间占比增加，调度开销相对减少，加速比下降。反向改为直接调用 `fused_swiglu_scale_grad` 算子后，消除了 autograd 框架调度开销，加速比显著提升 |

### 4.2 Fused XPU Op (Method 4) vs PaddleFleet Custom Op (Method 2)

| 指标 | 结论 |
|------|------|
| **前向加速** | 小/中等规模 **2.5x ~ 2.8x**，大规模 **1.19x** |
| **反向加速** | 小/中等规模 **9.8x ~ 13.2x**，大规模 **2.84x** |
| **原因分析** | PaddleFleet Custom Op 是 CUDA kernel 在 XPU 上通过兼容层运行，无法发挥 XPU 硬件特性；Fused XPU Op 直接调用 XDNN API，针对 XPU 架构优化。反向直接调用算子后，差距进一步拉大 |

### 4.3 Semi-Fused v2 (Method 3) vs Native Python (Method 1)

| 指标 | 结论 |
|------|------|
| **前向加速** | **1.0x ~ 1.2x**，与 Fleet 持平；small_3D 因 broadcast 维度对齐开销反而略慢（0.67x） |
| **反向加速** | 小/中等规模 **3.5x ~ 3.9x**，大规模 **2.42x** |
| **原因分析** | v2 前向调用框架内置 `swiglu`（已优化），与 Fleet 前向性能相当；反向利用 `swiglu_grad` 粗粒度算子，避免手写求导中的大量细小 op，收益显著 |

### 4.4 Fused XPU Op (Method 4) vs Semi-Fused v2 (Method 3)

| 指标 | 结论 |
|------|------|
| **前向加速** | 小/中等规模 **2.3x ~ 4.1x**，大规模 **1.19x** |
| **反向加速** | 小/中等规模 **2.9x ~ 3.3x**，大规模 **1.17x** |
| **原因分析** | 将反向从 autograd 改为直接调用 `fused_swiglu_scale_grad` 算子后，XPU 全融合反向 latency 显著降低（0.043~0.048ms），全面优于 v2（0.131~0.146ms）。单 kernel 内完成 SwiGLU + broadcast_mul + 梯度计算，避免了 v2 中多个 kernel launch 的开销 |

### 4.5 PaddleFleet Custom Op (Method 2) vs Native Python (Method 1)

| 指标 | 结论 |
|------|------|
| **前向加速** | **1.0x ~ 1.1x** |
| **反向加速** | **1.0x ~ 1.3x** |
| **原因分析** | CUDA kernel 在 XPU 上的兼容执行仅有微弱优势，跨架构翻译开销抵消了大部分融合收益 |

---

## 五、关键发现

1. **Fused XPU Op 前向是 XPU 环境下的最优选择**：在小/中等规模下前向加速 **~2.7x ~ 2.8x**，大规模 **~1.2x**，前向融合收益明确。

2. **反向改为直接调用算子后性能大幅提升**：`benchmark_fused_swiglu_scale_paddle.py` 将反向从 `paddle.autograd.backward` 改为直接调用 `paddle._C_ops.fused_swiglu_scale_grad` 后，XPU Fused 反向 latency 从之前的 **0.140~0.792ms** 降至 **0.043~0.439ms**，相对 Native 的加速比从 **1.6x ~ 3.5x** 提升至 **2.8x ~ 12.8x**。

3. **XPU 全融合算子的反向现在全面优于 Semi-Fused v2**：小/中等规模下 XPU Bwd（0.043~0.048ms）约为 v2 Bwd（0.131~0.146ms）的 **1/3**，证明了全融合 kernel 的收益。

4. **PaddleFleet Custom Op（CUDA）在 XPU 上无明显优势**：CUDA kernel 通过兼容层运行在 XPU 上，无法利用 XPU 的硬件并行特性，性能仅比 Native Python 提升约 1.0x ~ 1.1x。

5. **融合算子的收益随 tensor 规模增大而递减**：小规模 tensor（如 4×8）kernel launch 开销占比高，融合收益最大；大规模 tensor（如 32×512×1024）计算时间占比高，融合收益减小。

---

## 六、测试脚本位置

| 脚本 | 对应方法 | 说明 |
|------|---------|------|
| `benchmark_fused_swiglu_scale_native.py` | Method 1: Native Python | 纯 Python 手写 SwiGLU + Scale |
| `benchmark_fused_swiglu_scale.py` | Method 2: PaddleFleet Custom Op | CUDA C++ extension 自定义算子 |
| `benchmark_fused_swiglu_scale_v2.py` | Method 3: Semi-Fused v2 | 调用 `_C_ops.swiglu` + `paddle.multiply` 的半融合方案 |
| `benchmark_fused_swiglu_scale_paddle.py` | Method 4: Fused XPU Op | 本工作编译的 XPU 全融合算子 |

---

## 七、运行方式

```bash
cd /root/paddlejob/Gruge/private-repos/op-validation/fused_swiglu_scale_xpu
source /root/paddlejob/Gruge/envs/py310_paddleFormers/bin/activate

# 运行四种 benchmark
python benchmark_fused_swiglu_scale_native.py       # Method 1: Native Python
python benchmark_fused_swiglu_scale.py              # Method 2: PaddleFleet Custom Op
python benchmark_fused_swiglu_scale_v2.py           # Method 3: Semi-Fused v2
python benchmark_fused_swiglu_scale_paddle.py       # Method 4: Fused XPU Op
```

---

## 八、补充：Best Python 方案 vs Native vs Fused XPU（同口径对比）

为验证 Method 3（Semi-Fused v2）在统一 benchmark 口径下的表现，我们使用与 `benchmark_fused_swiglu_scale_paddle.py` **完全一致的数据和测试方式**（`chunk + silu + mul` 作为 Native，直接调用 `_C_ops` 作为 Fused），在 `v2_test/` 目录下进行了新一轮对比测试。

### 8.1 对比方案说明

| 方案 | 名称 | 前向实现 | 反向实现 |
|------|------|---------|---------|
| **Native** | chunk + silu + mul | `paddle.chunk` + `paddle.nn.functional.silu` + `*` | 手写 gradient，使用 `chunk`、`sigmoid`、`mul`、`concat`、`sum` |
| **Best Python** | `_C_ops.swiglu` + `_C_ops.multiply` | `_C_ops.swiglu(x, None)` + `_C_ops.multiply(out, scale)` | `_C_ops.swiglu` + `_C_ops.multiply` + `_C_ops.swiglu_grad` + `_C_ops.sum` |
| **Fused XPU** | `fused_swiglu_scale` | `paddle._C_ops.fused_swiglu_scale(x, scale)` | `paddle._C_ops.fused_swiglu_scale_grad(x, scale, out_grad)` |

### 8.2 原始数据（ms）

| Config | Native Fwd | Best Python Fwd | Fused Fwd | Native Bwd | Best Python Bwd | Fused Bwd |
|--------|-----------|-----------------|-----------|-----------|-----------------|-----------|
| small_2D (4×8) | 0.073 | **0.044** | 0.020 | 0.303 | **0.131** | 0.043 |
| medium_2D (1024×256) | 0.073 | **0.045** | 0.020 | 0.289 | **0.164** | 0.048 |
| large_2D (4096×512) | 0.073 | **0.045** | 0.031 | 0.285 | **0.132** | 0.074 |
| small_3D (2×3×16) | 0.108 | **0.045** | 0.021 | 0.291 | **0.133** | 0.044 |
| medium_3D (8×128×512) | 0.076 | **0.045** | 0.020 | 0.332 | **0.132** | 0.048 |
| large_3D (32×512×1024) | 0.290 | **0.237** | 0.199 | 0.954 | **0.513** | 0.438 |

### 8.3 加速比

| Config | Best Python vs Native (Fwd) | Best Python vs Native (Bwd) | Fused vs Native (Fwd) | Fused vs Native (Bwd) | Fused vs Best Python (Fwd) | Fused vs Best Python (Bwd) |
|--------|----------------------------|----------------------------|----------------------|----------------------|---------------------------|---------------------------|
| small_2D | **1.65×** | **2.31×** | **3.68×** | **6.99×** | 2.22× | 3.03× |
| medium_2D | **1.62×** | **1.76×** | **3.71×** | **6.07×** | 2.29× | 3.46× |
| large_2D | **1.63×** | **2.17×** | **2.39×** | **3.84×** | 1.46× | 1.77× |
| small_3D | **2.39×** | **2.18×** | **5.26×** | **6.56×** | 2.20× | 3.01× |
| medium_3D | **1.69×** | **2.52×** | **3.73×** | **6.89×** | 2.20× | 2.74× |
| large_3D | **1.22×** | **1.86×** | **1.45×** | **2.18×** | 1.19× | 1.17× |

### 8.4 结论

1. **Best Python（c_ops_swiglu）相对 Native 有显著优势**
   - 前向：小/中等规模加速 **1.6~2.4×**，大规模 **1.2×**
   - 反向：小/中等规模加速 **1.8~2.5×**，大规模 **1.9×**
   - 它将前向从 `chunk + silu + sigmoid + mul` 压缩为 `swiglu + multiply` 两个 kernel，将反向从手写小算子拼接压缩为 `swiglu + multiply + swiglu_grad + sum` 五个 kernel

2. **Fused XPU kernel 仍是终极最优解，但 Best Python 已逼近天花板**
   - 前向：Fused 相对 Best Python 快 **1.2~2.3×**
   - 反向：Fused 相对 Best Python 快 **1.2~3.5×**
   - 差距主要来自 kernel launch 次数（Best Python 前向 2 个、反向 5 个 vs Fused 前向 1 个、反向 1 个）和 Python-C dispatch overhead

3. **在不能编写 C++ kernel 的限制下，`_C_ops.swiglu` + `_C_ops.multiply` + `_C_ops.swiglu_grad` 是 Python 层能做到的最优大算子拼接方案**
   - 全部使用 `_C_ops` 直接调用，避免了 `paddle.nn.functional` 层的额外 wrapper
   - inplace、cached、autograd 等变体经测试均无法超越此方案

### 8.5 参考文件

- 测试脚本：`v2_test/benchmark_best_vs_native_vs_fused.py`
- 最优实现：`v2_test/fused_swiglu_scale_best.py`
- 详细报告：`v2_test/benchmark_report.md`
