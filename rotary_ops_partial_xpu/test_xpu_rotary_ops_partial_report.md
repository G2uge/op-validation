# XDNN 底层 RoPE 算子部分旋转位置编码支持性测试报告

## 1. 测试目的

验证 XDNN（XPU Deep Neural Network）底层库中两个旋转位置编码算子：

- `xpu::rotary_embedding_everytwo`（Neox 风格，相邻两维配对旋转）
- `xpu::rotary_embedding_half`（Half 风格，前后两半分别旋转）

是否支持**部分旋转位置编码**（Partial Rotary Position Embedding），即当传入的 `sin/cos` 的 `head_dim`（`freqs_shape` 的最后一个维度）小于输入 `q` 的 `head_dim`（`t_shape` 的最后一个维度）时的行为。

该测试行为对标 Paddle GPU 的 `fused_rotary_position_embedding` 实现（见 `paddle/phi/kernels/fusion/gpu/fused_rope_utils.h`）：
- 前 `pe_head_dim` 维应用旋转
- 后 `head_dim - pe_head_dim` 维保持不变（copy）

## 2. 测试环境

| 项目 | 说明 |
|------|------|
| 硬件 | XPU（通过 `xpu_set_device(0)` 初始化） |
| XDNN 库路径 | `Paddle/build/third_party/install/xpu/lib/libxpuapi.so` |
| XPU Runtime | `libxpurt.so` |
| 编译器 | g++ (C++11) |
| 测试代码 | `test_xpu_rotary_ops_partial.cc` |

## 3. 测试方法

通过独立的 C++ 程序直接调用 XDNN 算子，绕过 Paddle 框架层的任何预处理，向算子直接传入指定的 `q_shape` 与 `freqs_shape`，观察返回值及运行行为。

### 3.1 关键参数设定

- `batch_size = 1`
- `seq_len = 2`
- `num_heads = 1`
- `head_dim = 8`
- 数据类型：`float`

### 3.2 测试场景

1. **完整旋转**：`pe_head_dim == head_dim`（基准对照，校准 ref 公式）
2. **部分旋转前向**：`pe_head_dim = 4 < head_dim = 8`
3. **部分旋转反向**：`pe_head_dim = 4 < head_dim = 8`

### 3.3 CPU Reference 公式

#### rotary_embedding_half（前向）

```cpp
half = pe_head_dim / 2
for d in [0, half):
    out[d]       = q[d] * cos[d]       - q[d+half] * sin[d]
    out[d+half]  = q[d+half] * cos[d+half] + q[d] * sin[d+half]
for d in [pe_head_dim, head_dim):
    out[d] = q[d]   // copy unchanged
```

#### rotary_embedding_half（反向）

```cpp
half = pe_head_dim / 2
for d in [0, half):
    dq[d]      = do[d] * cos[d]       + do[d+half] * sin[d+half]
    dq[d+half] = do[d+half] * cos[d+half] - do[d] * sin[d]
for d in [pe_head_dim, head_dim):
    dq[d] = do[d]   // copy unchanged
```

#### rotary_embedding_everytwo（前向）

```cpp
even d : out[d] = q[d] * cos[d] - q[d+1] * sin[d]
odd  d : out[d] = q[d] * cos[d] + q[d-1] * sin[d]
for d in [pe_head_dim, head_dim):
    out[d] = q[d]   // copy unchanged
```

#### rotary_embedding_everytwo（反向）

```cpp
even d : dq[d] = do[d] * cos[d] + do[d+1] * sin[d+1]
odd  d : dq[d] = do[d] * cos[d] - do[d-1] * sin[d-1]
for d in [pe_head_dim, head_dim):
    dq[d] = do[d]   // copy unchanged
```

## 4. 测试结果

### 4.1 rotary_embedding_everytwo（Neox 风格）

| 场景 | pe_head_dim | head_dim | XPU 返回值 | 数值对比 | 结果说明 |
|------|-------------|----------|-----------|---------|----------|
| 完整旋转 | 8 | 8 | 0 (SUCCESS) | 与 ref **完全匹配** | 算子公式正确 |
| 部分旋转前向 | 4 | 8 | 1 (FAILED) | — | **不支持**，直接报错 |
| 部分旋转反向 | 4 | 8 | 1 (FAILED) | — | **不支持**，直接报错 |

### 4.2 rotary_embedding_half（Half 风格）

| 场景 | pe_head_dim | head_dim | XPU 返回值 | 数值对比 | 结果说明 |
|------|-------------|----------|-----------|---------|----------|
| 完整旋转 | 8 | 8 | 0 (SUCCESS) | 与 ref **完全匹配** | 算子公式正确 |
| 部分旋转前向 | 4 | 8 | 0 (SUCCESS) | 与 ref **完全匹配** | **支持**部分旋转 |
| 部分旋转反向 | 4 | 8 | 0 (SUCCESS) | 与 ref **完全匹配** | **支持**部分旋转反向 |

### 4.3 数值验证示例（half 部分旋转前向）

输入 `q = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]`，`pe_head_dim = 4`

| idx | q_in | ref (CPU) | xpu (XPU) | 匹配 |
|-----|------|-----------|-----------|------|
| 0 | 1.0 | 1.000 | 1.000 | Y |
| 1 | 1.1 | 1.085 | 1.085 | Y |
| 2 | 1.2 | 1.168 | 1.168 | Y |
| 3 | 1.3 | 1.249 | 1.249 | Y |
| 4 | 1.4 | 1.400 | 1.400 | Y |
| 5 | 1.5 | 1.511 | 1.511 | Y |
| 6 | 1.6 | 1.624 | 1.624 | Y |
| 7 | 1.7 | 1.739 | 1.739 | Y |

- 前 4 维（`pe_head_dim = 4`）按 half 风格旋转
- 后 4 维（`head_dim - pe_head_dim = 4`）保持不变

## 5. 结论

### 5.1 算子支持性总结

| 算子 | 前向部分旋转 | 反向部分旋转 | 是否支持 |
|------|-------------|-------------|---------|
| `rotary_embedding_everytwo` | ❌ 报错 | ❌ 报错 | **不支持** |
| `rotary_embedding_half` | ✅ 通过且数值正确 | ✅ 通过且数值正确 | **支持** |

### 5.2 核心发现

1. **`rotary_embedding_half` 原生支持部分旋转位置编码**
   - 当 `freqs_shape[-1] = pe_head_dim < head_dim` 时，算子正确执行：
     - **前 `pe_head_dim` 维**：按 half 风格旋转
     - **后 `head_dim - pe_head_dim` 维**：保持不变（copy）
   - 前向和反向（grad）均支持，数值结果与 CPU ref 完全匹配
   - 该行为与 Paddle GPU 的 `fused_rotary_position_embedding` 实现一致

2. **`rotary_embedding_everytwo` 不支持部分旋转位置编码**
   - 当 `freqs_shape[-1] != t_shape[-1]` 时，前向和反向均返回 `INVALID_PARAM`（ret=1）
   - 如果需要在 XPU 上使用 Neox 风格的部分旋转，必须在框架层把 `sin/cos` 填充到完整 `head_dim` 后再调用算子

3. **对 Paddle 框架层的启示**
   - `paddle/phi/kernels/fusion/xpu/fused_rope_utils.h` 中，当前对 `sin/cos` 的预处理（`GetSinCosByPassValue` / `GetSinCosByRotaryBase`）总是把 `sin/cos` 广播到完整 `head_dim`
   - 对于 `rotary_embedding_half`，这个预处理虽然安全，但如果需要优化内存/性能，可以直接传入 `pe_head_dim < head_dim` 的 `sin/cos`，由算子自身处理部分旋转
   - 对于 `rotary_embedding_everytwo`，由于底层算子不支持，框架层必须继续保持完整 `head_dim` 的预处理逻辑

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `test_xpu_rotary_ops_partial.cc` | C++ 单测源代码 |
| `test_xpu_rotary_ops_partial` | 编译后的可执行文件 |
| `test_xpu_rotary_ops_partial_report.md` | 本测试报告 |