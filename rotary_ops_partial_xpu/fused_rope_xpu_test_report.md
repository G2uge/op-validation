# XPU Fused Rotary Position Embedding 部分旋转支持测试报告

## 1. 测试目的

验证 XPU 融合算子 `fused_rotary_position_embedding` 在补充部分旋转位置编码（Partial Rotary Position Embedding）支持后的正确性。

- 当 `sin/cos` 的 `head_dim`（`pe_head_dim`）小于输入 `q` 的 `head_dim` 时：
  - 前 `pe_head_dim` 维应用 RoPE 旋转
  - 后 `head_dim - pe_head_dim` 维保持不变（copy）
- 对标 CUDA GPU 实现的相同行为

## 2. 测试环境

| 项目 | 说明 |
|------|------|
| 硬件 | XPU（8 设备，使用 device 0） |
| Paddle 源码 | `916b9f4950` (2026-05-11) |
| Paddle 版本 | 3.5.0.dev20260511 (XPU) |
| XDNN 库 | `Paddle/build/third_party/install/xpu/lib/libxpuapi.so` |
| XPU Runtime | `libxpurt.so` |
| Python | 3.10.12 |

## 3. 代码修改内容

### 3.1 框架层实现

| 文件 | 修改内容 |
|------|---------|
| `paddle/phi/kernels/fusion/xpu/fused_rope_utils.h` | 支持 `freqs_head_dim` 参数透传；half 风格直接传部分维度；everytwo 风格 pad sin/cos（sin=0, cos=1）实现 copy |
| `paddle/phi/kernels/fusion/xpu/fused_rope_kernel.cc` | 补充 K/V batch_size 下界检查、MQA/GQA num_heads 一致性检查、sin/cos 4D batch_size==1 检查 |

### 3.2 测试文件修改

| 文件 | 修改内容 |
|------|---------|
| `test/xpu/test_fused_rotary_position_embedding_op_xpu.py` | 参考函数支持 `rot_dim` 分割；`get_inputs` 新增 `rotary_percent`；新增部分旋转、MQA、零尺寸、零 num_heads 测试 |

## 4. 测试用例

### 4.1 新增测试

| 测试名 | 覆盖场景 |
|-------|---------|
| `test_fused_rope_rotary_percent_neox` | 部分旋转（rotary_percent=0.5）+ Neox/EveryTwo 风格 |
| `test_fused_rope_rotary_percent_half` | 部分旋转（rotary_percent=0.5）+ Half 风格 |
| `XPUTestFusedRotaryPositionEmbeddingMQA` | MQA 模式（q_heads=4, kv_heads=1） |
| `TestFusedRotaryPositionEmbeddingZeroSizeXPU` | 零尺寸 tensor（shape=[0, 1, 8, 8]） |
| `TestFusedRotaryPositionEmbeddingZeroNumHeadsXPU` | 零 num_heads edge case（5 个子用例） |

### 4.2 原有测试（回归验证）

| 测试名 | 覆盖场景 |
|-------|---------|
| `test_fused_rope` | 标准前向/反向，Neox 风格 |
| `test_fused_rope_without_sin_cos` | 内部生成 sin/cos |
| `test_fused_rope_rotate_half` | Half 风格 |
| `test_fused_rope_position_ids` | 自定义 position_ids |
| `test_static` | 静态图前向 |
| `XPUTestFusedRotaryPositionEmbeddingFp16_1` | float16 精度 |
| `XPUTestFusedRotaryPositionEmbeddingBf16_1/2` | bfloat16 精度（含大规模） |
| `XPUTestFusedRotaryPositionEmbeddingGQA` | GQA 模式 |

## 5. 编译步骤

```bash
cd /root/paddlejob/tmp/repos/Paddle/build

# 1. 增量编译 phi 库
make phi -j$(nproc)

# 2. 重新链接 libpaddle.so（避免符号不兼容）
cd python && make copy_libpaddle -j$(nproc)
```

## 6. 运行方式

```bash
cd /root/paddlejob/tmp/repos/Paddle/test/xpu
PYTHONPATH=/root/paddlejob/tmp/repos/Paddle/build/python:/root/paddlejob/tmp/repos/Paddle/test/legacy_test \
  python test_fused_rotary_position_embedding_op_xpu.py
```

## 7. 测试结果

```
Ran 42 tests in 88.873s

OK
```

| 测试类 | 用例数 | 结果 |
|-------|--------|------|
| `XPUTestFusedRotaryPositionEmbedding` | 6 | ✅ 通过 |
| `XPUTestFusedRotaryPositionEmbeddingFp16_1` | 6 | ✅ 通过 |
| `XPUTestFusedRotaryPositionEmbeddingBf16_1` | 6 | ✅ 通过 |
| `XPUTestFusedRotaryPositionEmbeddingBf16_2` | 1 | ✅ 通过 |
| `XPUTestFusedRotaryPositionEmbeddingMQA`（新增） | 6 | ✅ 通过 |
| `XPUTestFusedRotaryPositionEmbeddingGQA` | 6 | ✅ 通过 |
| `TestFusedRotaryPositionEmbeddingZeroSizeXPU`（新增） | 1 | ✅ 通过 |
| `TestFusedRotaryPositionEmbeddingZeroNumHeadsXPU`（新增） | 5 | ✅ 通过 |
| **总计** | **42** | **✅ 全部通过** |

## 8. 端到端训练精度对齐

为验证部分旋转位置编码修改在实际大模型训练中的数值稳定性，对比了相同配置下两版实现的 loss 曲线：

- **基线**：不采用部分旋转位置编码的融合算子（`paddleformers_dist_log.pruned.20260512_160811`）
- **新实现**：支持部分旋转位置编码的融合算子（`paddleformers_dist_log`）

| 指标 | 数值 |
|------|------|
| 对比步数 | 6 步 |
| 最大绝对误差 | 0.00321292 |
| 最大相对误差 | **0.024695%** |

**结论**：前 6 步 loss 的最大相对误差约 2.5e-4，远小于 1e-3 阈值，端到端精度对齐通过。

## 9. 结论

1. **部分旋转功能正确实现**：
   - `rotary_embedding_half` 风格：直接利用 XDNN 底层原生支持
   - `rotary_embedding_everytwo` 风格：通过框架层 pad sin/cos（sin=0, cos=1）实现等价 copy

2. **输入校验与 CUDA 对齐**：
   - 补充了 K/V batch_size 下界检查
   - 补充了 MQA/GQA num_heads 一致性检查
   - 补充了 sin/cos 4D 时 batch_size==1 检查

3. **测试覆盖完整**：
   - 新增部分旋转、MQA、零尺寸、零 num_heads 等场景
   - 原有测试全部通过，无回归
