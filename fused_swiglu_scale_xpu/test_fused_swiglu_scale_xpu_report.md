# fused_swiglu_scale XPU 算子精度验证报告

## 1. 测试目的

验证 Paddle PHI 层 `phi::fusion::FusedSwigluScaleKernel`（XPU 后端）的前向与反向计算精度，确保其数学行为与 CPU 参考实现一致。

## 2. 算子功能说明

`fused_swiglu_scale(x, scale)` 执行以下两步融合计算：

1. **SwiGLU**: 将输入 `x` 在最后一维分成两半，前半经过 SiLU 激活后与后半逐元素相乘：
   ```
   x1 = x[..., :hidden]
   x2 = x[..., hidden:]
   swiglu_out = SiLU(x1) * x2
              = x1 * sigmoid(x1) * x2
   ```
2. **Scale**: 将 `swiglu_out` 与 `scale` 做广播乘法：
   ```
   out = swiglu_out * scale
   ```

**反向传播** (`fused_swiglu_scale_grad`):
- `dx = swiglu_grad(x, dout * scale)`
- `dscale = sum(dout * swiglu_out, axis=-1)`

## 3. 测试环境

- **设备**: XPU (Kunlun)
- **XDNN 版本**: 与 Paddle build 同步
- **编译器**: g++ (C++11)
- **依赖库**: libxpuapi.so, libxpurt.so, libcudart.so

## 4. 测试方法

### 4.1 CPU 参考实现

- **前向**: 手写 Python 语义级 SwiGLU + broadcast scale
- **反向**: 手写 swiglu_grad（基于链式法则）+ reduce_sum

### 4.2 XPU 调用

直接调用 XDNN 底层 API，模拟 phi kernel 的执行流程：

- 前向: `xpu::swiglu` -> `xpu::broadcast_mul`
- 反向: `xpu::swiglu` -> `xpu::broadcast_mul` -> `xpu::swiglu_grad` -> `xpu::broadcast_mul` -> `xpu::reduce_sum`

### 4.3 测试场景

| 维度 | x shape | scale shape | 说明 |
|------|---------|-------------|------|
| 2D | [4, 8] | [4, 1] | 行级 scale |
| 2D | [4, 8] | [1, 4] | 列级 scale |
| 2D | [4, 8] | [4] | 1D scale broadcast |
| 3D | [2, 3, 16] | [2, 3, 8] | 3D 全匹配 |
| 3D | [2, 3, 16] | [1, 1, 8] | 3D 广播 |
| 4D | [2, 3, 4, 16] | [2, 1, 4, 8] | 4D 部分广播 |
| 2D | [1, 128] | [1, 64] | 单 batch |
| 2D | [1024, 256] | [1024, 1] | 大 batch |

## 5. 测试结果

```
Total tests: 13
Passed:      13
Failed:      0
```

### 5.1 前向精度

所有前向测试均通过，最大绝对误差在 `5.96e-08` 以内（float32 舍入误差级别）。

### 5.2 反向精度

所有反向测试均通过：
- `dx` 最大误差: `1.19e-07`
- `dscale` 最大误差: `2.15e-06`

误差来源为 float32 累加顺序差异，属于正常范围。

## 6. 结论

- `fused_swiglu_scale` XPU 算子的前向计算 **数学正确**
- `fused_swiglu_scale_grad` XPU 算子的反向计算 **数学正确**
- 支持多种 scale broadcast 模式
- 支持 2D/3D/4D 输入

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `test_fused_swiglu_scale_xpu.cc` | C++ 测试源码 |
| `build.sh` | 编译脚本 |
| `test_fused_swiglu_scale_xpu` | 可执行文件 |
| `test_fused_swiglu_scale_xpu_report.md` | 本报告 |

## 8. 编译与运行

```bash
cd /root/paddlejob/Gruge/private-repos/op-validation/fused_swiglu_scale_xpu
./build.sh
./test_fused_swiglu_scale_xpu
```
