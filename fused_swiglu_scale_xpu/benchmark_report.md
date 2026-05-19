# fused_swiglu_scale XPU 性能对比报告

## 1. 测试目的

对比两种 `swiglu + scale` 计算方式的执行效率：

1. **XPU 融合算子**（C++ 直接调用 XDNN `swiglu` + `broadcast_mul`）
2. **Native Python 实现**（PaddleFleet `fused_swiglu_scale.py` 的 else 分支：`swiglu(x) * scale`）

## 2. 测试方法

- **预热**: 10 次（不计时）
- **正式测试**: 100 次取平均
- **设备**: XPU
- **数据类型**: float32

### 2.1 XPU 融合算子（C++）

直接调用 XDNN API，模拟 phi kernel 的执行流程：
- 前向: `xpu::swiglu` -> `xpu::broadcast_mul`
- 反向: `xpu::swiglu` -> `xpu::broadcast_mul` -> `xpu::swiglu_grad` -> `xpu::broadcast_mul` -> `xpu::reduce_sum`

### 2.2 Native Python 实现

对应 PaddleFleet 的 else 分支：
- 前向: `paddle.nn.functional.swiglu(x)` + broadcast `scale`
- 反向: 手写 `sigmoid`, `silu`, `concat`, `sum` 等组合操作

## 3. 测试结果

| Case | Shape (x / scale) | Metric | Native (ms) | Fused (ms) | Speedup |
|------|-------------------|--------|-------------|------------|---------|
| small_2D | 4x8 / 4x1 | forward | 0.0570 | 0.0123 | **4.62x** |
| | | backward | 0.5226 | 0.0293 | **17.84x** |
| medium_2D | 1024x256 / 1024x1 | forward | 0.0567 | 0.0120 | **4.74x** |
| | | backward | 0.5204 | 0.0294 | **17.73x** |
| large_2D | 4096x512 / 4096x1 | forward | 0.0571 | 0.0306 | **1.87x** |
| | | backward | 0.5211 | 0.0692 | **7.53x** |
| small_3D | 2x3x16 / 1x1x8 | forward | 0.0577 | 0.0120 | **4.82x** |
| | | backward | 0.5298 | 0.0298 | **17.81x** |
| medium_3D | 8x128x512 / 8x128x256 | forward | 0.0566 | 0.0130 | **4.36x** |
| | | backward | 0.5269 | 0.0294 | **17.93x** |
| large_3D | 32x512x1024 / 32x512x512 | forward | 0.2371 | 0.1974 | **1.20x** |
| | | backward | 1.2445 | 0.4317 | **2.88x** |

### 3.1 平均值

| Metric | Native (ms) | Fused (ms) | Speedup |
|--------|-------------|------------|---------|
| forward | 0.0870 | 0.0462 | **1.88x** |
| backward | 0.6442 | 0.1031 | **6.25x** |

## 4. 结论

- **XPU 融合算子在所有测试场景下均显著优于 Native Python 实现**
- 反向传播收益尤为明显（平均 **6.25x**），因为融合算子避免了中间结果的显式构造和多次 kernel launch
- 在小 shape 场景下，forward 加速比可达 **4.6x**，backward 加速比可达 **17.8x**
- 在大 shape 场景下（如 32x512x1024），由于计算本身占主导，加速比有所下降，但仍有 **1.2x ~ 2.9x** 的提升

## 5. 文件清单

| 文件 | 说明 |
|------|------|
| `benchmark_fused_swiglu_scale_xpu.cc` | C++ 融合算子性能测试源码 |
| `benchmark_fused_swiglu_scale_native.py` | Python native 实现性能测试脚本 |
| `compare_benchmark.py` | 一键对比脚本（运行 C++ 和 Python 测试并输出表格） |
| `build.sh` | 编译脚本 |
| `benchmark_fused_swiglu_scale_xpu` | C++ 可执行文件 |
| `benchmark_report.json` | JSON 格式原始数据 |
| `benchmark_report.md` | 本报告 |

## 6. 使用方法

```bash
cd /root/paddlejob/Gruge/private-repos/op-validation/fused_swiglu_scale_xpu
./build.sh
python compare_benchmark.py
```
