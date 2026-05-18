# SwiGLU Backward XPU 精度回归测试报告

> **对应代码**：`src/paddlefleet/fusions/fused_bias_swiglu.py` 中 XPU 分支
> ```python
> elif paddle.is_compiled_with_xpu():
>     dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
>     return dx
> ```
>
> **测试文件**：[test_swiglu_grad_xpu_accuracy.py](test_swiglu_grad_xpu_accuracy.py)
>
> **测试日期**：2026-05-18

---

## 1. 测试目的

验证 `paddle._C_ops.swiglu_grad` 在 XPU 后端上的输出与纯 Paddle 参考实现（分解为 `split / sigmoid / concat` 等基础算子）在数值上是否等价，以满足 checklist **A5（Fusions 变更需证明与非融合路径数值一致）** 的要求。

---

## 2. 测试环境

| 项目 | 值 |
|------|-----|
| Paddle 版本 | 3.5.0.dev20260511 |
| XPU 编译状态 | `True` |
| 当前设备 | `xpu:0` |
| Python 路径 | `/root/paddlejob/tmp/paddle/bin/python` |
| `paddlefleet` 源码路径 | `/root/paddlejob/tmp/repos/PaddleFleet/src`（可编辑安装） |
| 运行时间 | 2026-05-18 |
| 测试框架 | `unittest` |

---

## 3. 参考实现说明

参考实现 `_reference_swiglu_grad` 完全基于 Paddle 基础 API，不依赖任何融合算子：

```python
y1, y2 = paddle.split(y, 2, axis=-1)
sigmoid_y1 = F.sigmoid(y1)
silu_y1 = y1 * sigmoid_y1

grad_y1 = g * y2 * sigmoid_y1 * (1.0 + y1 * (1.0 - sigmoid_y1))
grad_y2 = g * silu_y1

dx = paddle.concat([grad_y1, grad_y2], axis=-1)
```

该参考实现与 XPU 融合算子 `paddle._C_ops.swiglu_grad(y, None, g)` 的数学语义完全一致。

---

## 4. 测试用例覆盖

| 编号 | 测试方法 | 输入 shape `y` | 输入 shape `g` | 数据类型 | 误差阈值 |
|------|---------|----------------|---------------|---------|---------|
| 1 | `test_swiglu_grad_xpu_float32_2d` | `[4, 8]` | `[4, 4]` | float32 | `rtol=1e-5, atol=1e-5` |
| 2 | `test_swiglu_grad_xpu_float16_2d` | `[4, 8]` | `[4, 4]` | float16 | `rtol=1e-3, atol=1e-3` |
| 3 | `test_swiglu_grad_xpu_float32_3d` | `[2, 3, 8]` | `[2, 3, 4]` | float32 | `rtol=1e-5, atol=1e-5` |
| 4 | `test_swiglu_grad_xpu_float16_3d` | `[2, 3, 8]` | `[2, 3, 4]` | float16 | `rtol=1e-3, atol=1e-3` |
| 5 | `test_swiglu_grad_xpu_large_dim` | `[8, 4096]` | `[8, 2048]` | float32 | `rtol=1e-5, atol=1e-5` |
| 6 | `test_swiglu_grad_xpu_edge_broadcast` | `[1, 8]` | `[1, 4]` | float32 | `rtol=1e-5, atol=1e-5` |

### 覆盖维度说明
- **2D / 3D**：覆盖 `bias_swiglu_impl` 中直接调用和 reshape 后的两种典型输入维度。
- **float32 / float16**：覆盖训练推理中常见的两种精度。
- **大 hidden dim**：`[8, 4096]` 对应大模型 MLP 中 SwiGLU 的真实规模。
- **batch=1 边界**：验证 broadcast/边缘场景的正确性。

---

## 5. 测试结果

### 5.1 总体结果

```
Ran 6 tests in 0.022s
OK
```

**全部 6 个测试用例通过，无失败、无跳过。**

### 5.2 各用例数值差异详情

| 用例标识 | max_diff | mean_diff | 结论 |
|---------|----------|-----------|------|
| `fp32_2d` | `5.96e-08` | `3.09e-09` | 通过 |
| `fp32_3d` | `1.19e-07` | `3.71e-09` | 通过 |
| `fp32_large` | `9.54e-07` | `5.82e-09` | 通过 |
| `fp32_1batch` | `1.49e-08` | `2.79e-09` | 通过 |
| `fp16_2d` | `1.95e-03` | `1.13e-04` | 通过 |
| `fp16_3d` | `9.77e-04` | `8.37e-05` | 通过 |

### 5.3 结果分析

- **float32**：max_diff 量级在 `1e-07 ~ 1e-08` 之间，远低于 `rtol=1e-5` 的阈值，说明 XPU 融合实现与参考实现几乎完全对齐。
- **float16**：max_diff 量级在 `1e-03` 左右，处于 `rtol=1e-3` 阈值范围内。float16 的误差略大是由低精度计算本身的舍入特性导致，属于正常现象。
- **大维度场景**（`[8, 4096]`）：max_diff 为 `9.54e-07`，依然在 float32 安全范围内，说明规模放大不会引入累积误差。

---

## 6. 结论

`paddle._C_ops.swiglu_grad` 在 XPU 后端上的输出与纯 Paddle 参考实现**数值等价**，满足 checklist **A5** 的要求。该 XPU 分支可以安全合入。

---

## 7. 复现命令

在已编译 XPU 的 Paddle 虚拟环境中执行：

```bash
cd /root/paddlejob/tmp/test
/root/paddlejob/tmp/paddle/bin/python test_swiglu_grad_xpu_accuracy.py -v
```

或使用虚拟环境激活后执行：

```bash
source /root/paddlejob/tmp/paddle/bin/activate
cd /root/paddlejob/tmp/test
python test_swiglu_grad_xpu_accuracy.py -v
```

若当前环境未编译 XPU，则全部测试会自动跳过（`skipped=6`），请在 XPU 机器上手动运行并更新本报告。
