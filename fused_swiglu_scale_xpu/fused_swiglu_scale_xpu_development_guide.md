# fused_swiglu_scale XPU 融合算子开发全流程总结

## 一、算子功能说明

`fused_swiglu_scale` 是 SwiGLU 激活与 Scale 乘法的融合算子。

- **输入**：`x`（最后一维为偶数，会被拆分为 gate 和 value）、`scale`
- **计算**：`out = SiLU(gate) * value * scale`
- **反向**：输出 `x_grad`、`scale_grad`

融合的意义在于将多个 XDNN API 调用合并为一次 kernel 调用，减少内存访问和调度开销。

---

## 二、需要修改的文件及原因

### 1. Kernel 实现

**文件**：`paddle/phi/kernels/fusion/xpu/fused_swiglu_scale_kernel.cc`

**内容**：
- 前向：`xpu::swiglu` → `xpu::broadcast_mul`
- 反向：`xpu::swiglu` → `xpu::broadcast_mul` → `xpu::swiglu_grad` → `xpu::broadcast_mul` → `xpu::reduce_sum`
- `PD_REGISTER_KERNEL` 注册算子，支持 float、float16、bfloat16

**注意事项**：
- `xpu::swiglu_grad` 必须显式写模板参数 `<XPUType>`，不能隐式推导（`nullptr` 无法推导 `const T*`）
- `scale_shape` 需要在**前面**补 1（`insert(begin, 1)`），不能后面补，否则 broadcast 语义错误
- 反向 `dscale` 的 `reduce_sum` 应该对 batch 维度求和，而非最后一维

### 2. InferMeta（形状推导）

**文件**：
- `paddle/phi/infermeta/fusion.h`（声明）
- `paddle/phi/infermeta/fusion.cc`（实现）

**内容**：
- `FusedSwigluScaleInferMeta`：输出最后一维减半（`2H → H`）
- `FusedSwigluScaleGradInferMeta`：`x_grad` 共享 `x` 的 meta，`scale_grad` 共享 `scale` 的 meta

**为什么需要 InferMeta**：
Paddle 在执行算子前需要先知道输出张量的 shape/dtype/layout，InferMeta 在编译期/执行期提供这些信息，不依赖实际数据。

### 3. YAML 配置

**文件**：
- `paddle/phi/ops/yaml/fused_ops.yaml`（前向算子定义）
- `paddle/phi/ops/yaml/fused_backward.yaml`（反向算子定义）

**关键字段**：
```yaml
- op : fused_swiglu_scale
  args : (Tensor x, Tensor scale)
  output : Tensor(out)
  infer_meta :
    func : FusedSwigluScaleInferMeta    # 必须与 InferMeta 函数名一致
  kernel :
    func : fused_swiglu_scale
    data_type : x
  backward : fused_swiglu_scale_grad
  support_dygraph_mode : true            # 必须设置，否则 eager 代码生成器跳过
```

**为什么需要 YAML**：
Paddle 的代码生成器（`api_gen.py`、`backward_api_gen.py`、`eager_gen.py`）读取 YAML，自动生成 C++ API 声明、反向 API、Eager 动态图函数。YAML 是算子与框架的"契约"。

### 4. 代码生成器产物（编译时自动/手动同步）

**文件**：
- `paddle/phi/api/include/fused_api.h`
- `paddle/phi/api/backward/fused_backward_api.h`
- `paddle/phi/api/backward/fused_backward_api_base.h`
- `paddle/fluid/eager/api/generated/.../dygraph_functions.cc`
- `paddle/fluid/eager/api/generated/.../dygraph_grad_functions.cc`

**为什么需要同步**：
CMake 在 `cmake ..` 阶段运行代码生成器，产出 `.tmp` 文件，再通过 `copy_if_different` 同步到正式文件。如果：
- 修改了 YAML 但没有重新 `cmake ..`
- 或者 `.tmp` 与正式文件时间戳混乱

就会导致正式文件中**缺失新算子的 API 声明**，编译报错。

**手动同步命令**：
```bash
# 运行生成器
python3 paddle/phi/api/generator/api_gen.py ...
python3 paddle/phi/api/generator/backward_api_gen.py ...
python3 paddle/fluid/eager/auto_code_generator/generator/eager_gen.py ...

# 同步 .tmp → 正式文件
cmake -E copy_if_different ...
```

---

## 三、编译流程与注意事项

### 编译步骤

```bash
cd /root/paddlejob/tmp/repos/Paddle/build

# Step 1: 重新配置（如果新增了 .cc 文件）
cmake .. -DWITH_XPU=ON

# Step 2: 编译 phi 库（InferMeta + Kernel）
make -j$(nproc) phi

# Step 3: 编译并链接 libpaddle.so（C++ API + Eager）
make -j$(nproc) copy_libpaddle

# Step 4: whl 打包
cd /tmp
rm -rf wheel_extracted
unzip -q /path/to/existing.whl -d wheel_extracted
cp build/python/paddle/base/libpaddle.so wheel_extracted/paddle/base/
cp build/paddle/phi/libphi_core.so wheel_extracted/paddle/libs/
cp build/paddle/phi/libphi.so wheel_extracted/paddle/libs/
python3 -m wheel pack wheel_extracted --dest-dir /path/to/output/

# Step 5: 安装
pip install --force-reinstall --no-deps /path/to/paddlepaddle_xpu-*.whl
```

### 编译注意事项

| 问题 | 原因 | 解决 |
|------|------|------|
| **运行时 `kernel not found`** | CMake `file(GLOB)` 缓存未收录新增 `.cc` | 新增文件后必须重新 `cmake ..` |
| **`is not a member of paddle::experimental`** | `fused_api.h` 未同步 | 运行 `api_gen.py` + `copy_if_different` |
| **`std::tuple` + `enable_if<false>`** | `fused_backward_api.h` 缺失声明 | 检查并同步 `fused_backward_api.h` |
| **`too many arguments`** | `fused_backward_api_base.cc` 未同步 | 运行 `backward_api_gen.py` + 同步 |
| **`undefined symbol`** at runtime | whl 中 `libphi_core.so` 未更新 | 打包时同时替换 `libpaddle.so`、`libphi_core.so`、`libphi.so` |

### 编译产物依赖关系

```
YAML (fused_ops.yaml / fused_backward.yaml)
    ↓
InferMeta (fusion.h / fusion.cc)
    ↓
Kernel (fused_xxx_kernel.cc)
    ↓
代码生成器 (api_gen.py / backward_api_gen.py / eager_gen.py)
    ↓
.tmp 文件 → copy_if_different → 正式头文件/源文件
    ↓
make phi → libphi_core.so (Kernel + InferMeta)
    ↓
make copy_libpaddle → libpaddle.so (链接 phi + pybind + eager)
    ↓
whl 打包
```

---

## 四、测试文件编写

### 文件位置

`Paddle/test/xpu/test_fused_swiglu_scale_op_xpu.py`

### 核心设计

#### 1. 参考实现（numpy）

```python
def ref_swiglu_scale(x, scale):
    hidden = x.shape[-1] // 2
    gate = x[..., :hidden]
    val = x[..., hidden:]
    silu_gate = gate / (1.0 + np.exp(-gate))
    swiglu = silu_gate * val
    return swiglu * scale
```

#### 2. 梯度测试方式

融合算子通常为 `dygraph-only`（`support_dygraph_mode: true`），**不支持旧静态图的 `check_grad_with_place`**。因此采用 Paddle 融合算子的标准做法：

```python
def setUp(self):
    self.op_type = 'fused_swiglu_scale'
    self.__class__.no_need_check_grad = True   # 关键：跳过旧框架的梯度检查
    ...

def test_check_grad(self):
    paddle.disable_static()

    # 参考实现（Paddle 标准 op）
    x_ref = paddle.to_tensor(x_np)
    gate, val = paddle.chunk(x_ref, 2, axis=-1)
    out_ref = F.silu(gate) * val * scale
    paddle.autograd.backward([out_ref], [dout])

    # 融合算子
    x_custom = paddle.to_tensor(x_np)
    out_custom = paddle._C_ops.fused_swiglu_scale(x_custom, scale)
    paddle.autograd.backward([out_custom], [dout])

    # 比对梯度
    np.testing.assert_allclose(x_custom.grad.numpy(), x_ref.grad.numpy())
    np.testing.assert_allclose(scale_custom.grad.numpy(), scale_ref.grad.numpy())
```

**为什么设置 `no_need_check_grad = True`**：

旧框架的 `tearDownClass` 会检查 float16 测试是否执行过 `check_grad_with_place`。如果不设置这个标志，且没有调用 `check_grad_with_place`（因为会报 `NotImplementedError`），`tearDownClass` 会报错：
```
AssertionError: This test of fused_swiglu_scale op needs check_grad.
```

这是 Paddle 新融合算子的**标准做法**，不代表算子有缺陷。

#### 3. 测试覆盖

| 变体 | x_shape | scale_shape | 测试点 |
|------|---------|-------------|--------|
| Base | `[2, 8]` | `[4]` | 标准 2D + 1D scale |
| Op1 | `[4, 128]` | `[64]` | 中等规模 |
| Op2 | `[2, 4, 256]` | `[128]` | 3D 输入 |
| Op3 | `[8, 4096]` | `[2048]` | 大 hidden size |
| Op4 | `[1, 16]` | `[8]` | batch=1 边界 |
| Op5 | `[8, 2]` | `[1]` | 最小 hidden=2 |

#### 4. dtype 覆盖

```python
support_types = get_xpu_op_support_types('fused_swiglu_scale')
for stype in support_types:
    create_test_class(globals(), XPUTestFusedSwigluScaleOp, stype)
```

动态生成 float32、float16、bfloat16 三类测试。

---

## 五、测试结果

### 第一轮：基础精度测试

```
Ran 36 tests in 0.313s
OK
```

| 测试类别 | 数量 | 结果 |
|---------|------|------|
| `test_check_output`（前向精度） | 18 | 全部通过 |
| `test_check_grad`（反向梯度） | 18 | 全部通过 |

### 第二轮：扩展鲁棒性测试（新增）

```
Ran 44 tests in 0.302s
OK
```

在基础 36 个测试之上，新增了 8 个独立测试类，覆盖以下维度：

| 测试类别 | 测试类 | 数量 | 结果 |
|---------|--------|------|------|
| **空输入处理** | `TestFusedSwigluScaleZeroSize` | 1 | ✅ 通过 |
| **非法输入校验** | `TestFusedSwigluScaleError` | 2 | ✅ 通过 |
| **静态图兼容性** | `TestFusedSwigluScaleStatic` | 1 | ✅ 通过 |
| **AMP / 混合精度** | `TestFusedSwigluScaleMixedPrecision` | 1 | ✅ 通过 |
| **极端值稳定性** | `TestFusedSwigluScaleNumerical` | 2 | ✅ 通过 |
| **连续调用/确定性** | `TestFusedSwigluScaleStress` | 1 | ✅ 通过 |

### 完整测试覆盖矩阵

| 维度 | 数量 | 状态 |
|------|------|------|
| 前向精度（多 shape × 多 dtype） | 18 | ✅ |
| 反向梯度（dygraph 比对） | 18 | ✅ |
| 空输入 / zero-size | 1 | ✅ |
| 非法输入 / Error handling | 2 | ✅ |
| 静态图兼容性 | 1 | ✅ |
| AMP bf16 + fp32 loss | 1 | ✅ |
| 极端值稳定性（大值/接近0） | 2 | ✅ |
| 确定性 / 连续调用 | 1 | ✅ |
| **总计** | **44** | **全部通过** |

**Dtype 覆盖**：float32 ✅、float16 ✅、bfloat16 ✅

**Shape 覆盖**：2D、3D、batch=1、大 hidden（4096）、最小 hidden（2）全部通过

---

## 六、关键踩坑记录

### 1. CMake GLOB 缓存陷阱

**现象**：`make phi` 成功，但运行时 `kernel not found`。

**原因**：`paddle/phi/kernels/CMakeLists.txt` 使用 `file(GLOB)` 扫描 `.cc` 文件，只在 `cmake ..` 时执行一次。新增 `fused_swiglu_scale_kernel.cc` 后没有重新 `cmake ..`，文件未被收录。

**解决**：新增 `.cc` 文件后**必须**重新运行 `cmake ..`。

### 2. `std::tuple` 报错误导性极强

**现象**：`dygraph_grad_functions.cc` 报 `std::tuple::tuple(<brace-enclosed initializer list>)` + `enable_if<false>`。

**原因**：`fused_backward_api.h` 未同步，缺失 `fused_swiglu_scale_grad` 声明。编译器模板推导失败，错误最终落在 `std::tuple` 构造函数上。

**解决**：同步 `fused_backward_api.h`，不是修改 C++ 语法。

### 3. whl 打包只替换 `libpaddle.so` 不够

**现象**：安装后 `import paddle` 报 `undefined symbol`。

**原因**：`libphi_core.so`（包含 kernel 实现）没有替换进 whl，旧 so 没有新算子符号。

**解决**：打包时同时替换 `libpaddle.so`、`libphi_core.so`、`libphi.so`。

### 4. `xpu::swiglu_grad` 模板参数

**现象**：编译报 `no matching function for call`，`mismatched types 'const T*' and 'std::nullptr_t'`。

**原因**：`xpu::swiglu_grad(ctx, x_ptr, nullptr, ...)` 没有显式写 `<XPUType>`，编译器无法从 `nullptr` 推导 `T`。

**解决**：显式写 `xpu::swiglu_grad<XPUType>(...)`。

### 5. broadcast 维度方向

**现象**：`scale_shape` 从后面补 1（`push_back`）导致 broadcast 语义错误。

**原因**：XDNN 的 `broadcast_mul` 遵循 numpy 规则，低维张量需要在**前面**补 1 才能和高维张量对齐。

**解决**：`scale_shape.insert(scale_shape.begin(), count, 1)`。

---

## 七、参考文件清单

| 文件 | 作用 |
|------|------|
| `paddle/phi/kernels/fusion/xpu/fused_swiglu_scale_kernel.cc` | Kernel 实现 |
| `paddle/phi/infermeta/fusion.h` / `fusion.cc` | InferMeta |
| `paddle/phi/ops/yaml/fused_ops.yaml` | 前向 YAML |
| `paddle/phi/ops/yaml/fused_backward.yaml` | 反向 YAML |
| `Paddle/test/xpu/test_fused_swiglu_scale_op_xpu.py` | 测试文件 |
| `Paddle/test/xpu/op_test_xpu.py` | XPU 测试基类 |
| `Paddle/test/xpu/get_test_cover_info.py` | 测试辅助工具 |
