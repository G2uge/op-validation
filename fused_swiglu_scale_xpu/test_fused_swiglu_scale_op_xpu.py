# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core


def ref_swiglu_scale(x, scale):
    """Reference implementation of fused_swiglu_scale using numpy.

    SwiGLU: split x along last dim into gate and value,
    compute SiLU(gate) * value, then multiply by scale.
    """
    hidden = x.shape[-1] // 2
    gate = x[..., :hidden]
    val = x[..., hidden:]
    # SiLU(x) = x * sigmoid(x)
    silu_gate = gate / (1.0 + np.exp(-gate))
    swiglu = silu_gate * val
    return swiglu * scale


def ref_swiglu_scale_grad(x, scale, out_grad):
    """Reference backward implementation using numpy/Paddle ops."""
    hidden = x.shape[-1] // 2
    gate = x[..., :hidden]
    val = x[..., hidden:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    silu = gate * sig
    swiglu = silu * val

    # d_scale = sum(out_grad * swiglu, over batch axes)
    d_scale = np.sum(out_grad * swiglu, axis=tuple(range(out_grad.ndim - 1)))

    # Backprop through scale multiply
    d_u = out_grad * scale

    # Backprop through SwiGLU
    d_val = d_u * silu
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))
    d_x = np.concatenate([d_gate, d_val], axis=-1)

    return d_x, d_scale


class XPUTestFusedSwigluScaleOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fused_swiglu_scale'
        self.use_dynamic_create_class = False

    class TestFusedSwigluScaleBase(XPUOpTest):
        def setUp(self):
            self.op_type = 'fused_swiglu_scale'
            # NOTE: This op is dygraph-oriented. We manually check grad
            # in dygraph mode instead of using framework's static graph
            # check_grad_with_place, which requires backward registration
            # in static mode.
            self.__class__.no_need_check_grad = True
            self.init_dtype_type()
            self.init_datas_shape_and_attrs()

            x_np = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
            scale_np = np.random.uniform(
                0.5, 1.5, self.scale_shape
            ).astype(self.dtype)
            out_np = ref_swiglu_scale(x_np, scale_np).astype(self.dtype)

            self.inputs = {'x': x_np, 'scale': scale_np}
            self.outputs = {'out': out_np}
            self.attrs = {}

        def init_dtype_type(self):
            self.dtype = self.in_type
            self.atol = 1e-4
            if self.dtype == np.float16:
                self.atol = 1e-3
            elif self.dtype == np.uint16:
                # bfloat16 is stored as uint16 in numpy
                self.atol = 2e-2

        def init_datas_shape_and_attrs(self):
            # x: [B, 2H], scale: [H], out: [B, H]
            self.x_shape = [2, 8]
            self.scale_shape = [4]

        def test_check_output(self):
            self.check_output_with_place(core.XPUPlace(0), atol=self.atol)

        def test_check_grad(self):
            # NOTE: This op is dygraph-oriented. Static graph grad check
            # may fail due to backward registration limitations.
            # We test gradient correctness via dygraph instead.
            self._test_dygraph_grad()

        def _test_dygraph_grad(self):
            """Dygraph gradient test against reference implementation."""
            x_np = self.inputs['x']
            scale_np = self.inputs['scale']
            dout_np = np.random.uniform(-1, 1, self.outputs['out'].shape).astype(
                'float32'
            )

            paddle.disable_static()

            # Reference: Paddle standard ops
            x_ref = paddle.to_tensor(x_np.astype('float32'))
            scale_ref = paddle.to_tensor(scale_np.astype('float32'))
            x_ref.stop_gradient = False
            scale_ref.stop_gradient = False

            gate_ref, val_ref = paddle.chunk(x_ref, chunks=2, axis=-1)
            silu_ref = gate_ref * paddle.nn.functional.sigmoid(gate_ref)
            swiglu_ref = silu_ref * val_ref
            out_ref = swiglu_ref * scale_ref

            dout_ref = paddle.to_tensor(dout_np)
            paddle.autograd.backward([out_ref], [dout_ref])

            # Custom op
            x_custom = paddle.to_tensor(x_np.astype('float32'))
            scale_custom = paddle.to_tensor(scale_np.astype('float32'))
            x_custom.stop_gradient = False
            scale_custom.stop_gradient = False

            out_custom = paddle._C_ops.fused_swiglu_scale(x_custom, scale_custom)
            paddle.autograd.backward([out_custom], [dout_ref])

            # Compare
            np.testing.assert_allclose(
                x_custom.grad.numpy(),
                x_ref.grad.numpy(),
                rtol=self.atol,
                atol=self.atol,
                err_msg=f"x_grad mismatch: shape={self.x_shape}, scale={self.scale_shape}",
            )
            np.testing.assert_allclose(
                scale_custom.grad.numpy(),
                scale_ref.grad.numpy(),
                rtol=self.atol,
                atol=self.atol,
                err_msg=f"scale_grad mismatch: shape={self.x_shape}, scale={self.scale_shape}",
            )

            paddle.enable_static()

    class TestFusedSwigluScaleOp1(TestFusedSwigluScaleBase):
        def init_datas_shape_and_attrs(self):
            self.x_shape = [4, 128]
            self.scale_shape = [64]

    class TestFusedSwigluScaleOp2(TestFusedSwigluScaleBase):
        def init_datas_shape_and_attrs(self):
            # 3D input
            self.x_shape = [2, 4, 256]
            self.scale_shape = [128]

    class TestFusedSwigluScaleOp3(TestFusedSwigluScaleBase):
        def init_datas_shape_and_attrs(self):
            # Large hidden size
            self.x_shape = [8, 4096]
            self.scale_shape = [2048]

    class TestFusedSwigluScaleOp4(TestFusedSwigluScaleBase):
        def init_datas_shape_and_attrs(self):
            # Batch size = 1
            self.x_shape = [1, 16]
            self.scale_shape = [8]

    class TestFusedSwigluScaleOp5(TestFusedSwigluScaleBase):
        def init_datas_shape_and_attrs(self):
            # Small hidden size (minimum viable: 2)
            self.x_shape = [8, 2]
            self.scale_shape = [1]


class TestFusedSwigluScaleZeroSize(unittest.TestCase):
    """Test zero-size (empty batch) input handling."""

    def test_zero_size(self):
        paddle.disable_static()
        x_np = np.random.randn(0, 8).astype('float32')
        scale_np = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
        x = paddle.to_tensor(x_np)
        scale = paddle.to_tensor(scale_np)
        out = paddle._C_ops.fused_swiglu_scale(x, scale)
        self.assertEqual(out.shape, [0, 4])
        self.assertEqual(out.dtype, x.dtype)
        paddle.enable_static()


class TestFusedSwigluScaleError(unittest.TestCase):
    """Test error handling for invalid inputs."""

    def test_odd_last_dim(self):
        # Last dimension of x must be even for SwiGLU split
        paddle.disable_static()
        x = paddle.to_tensor(np.random.randn(2, 7).astype('float32'))
        scale = paddle.to_tensor([1.0, 2.0, 3.0])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle._C_ops.fused_swiglu_scale(x, scale)
        paddle.enable_static()

    def test_invalid_dtype(self):
        # Only float16, bfloat16, float32 are supported
        paddle.disable_static()
        x = paddle.to_tensor(np.random.randn(2, 8).astype('int32'))
        scale = paddle.to_tensor([1, 2, 3, 4])
        with self.assertRaises((TypeError, RuntimeError)):
            paddle._C_ops.fused_swiglu_scale(x, scale)
        paddle.enable_static()


class TestFusedSwigluScaleStatic(unittest.TestCase):
    """Test static graph (declarative mode) compatibility."""

    def test_static_shape(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(
                name='x', shape=[None, 8], dtype='float32'
            )
            scale = paddle.static.data(
                name='scale', shape=[4], dtype='float32'
            )
            out = paddle._C_ops.fused_swiglu_scale(x, scale)
            # Verify output shape inference at compile time
            self.assertEqual(out.shape[-1], 4)
        paddle.disable_static()


class TestFusedSwigluScaleMixedPrecision(unittest.TestCase):
    """Test bfloat16 path with fp32 accumulation (common in AMP training)."""

    def test_bf16_with_fp32_loss(self):
        paddle.disable_static()
        x = paddle.randn([4, 128], dtype='bfloat16')
        x.stop_gradient = False
        scale = paddle.randn([64], dtype='bfloat16')
        scale.stop_gradient = False

        out = paddle._C_ops.fused_swiglu_scale(x, scale)
        # Cast to fp32 for loss computation (typical AMP pattern)
        loss = out.astype('float32').sum()
        loss.backward()

        # Verify no crash and grads exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(scale.grad)
        paddle.enable_static()


class TestFusedSwigluScaleNumerical(unittest.TestCase):
    """Test numerical stability under extreme values."""

    def test_large_values(self):
        paddle.disable_static()
        # Large positive and negative values
        x_np = np.array([[100.0, 100.0, -100.0, -100.0]], dtype='float32')
        x = paddle.to_tensor(x_np)
        scale = paddle.to_tensor([1.0, 1.0])
        out = paddle._C_ops.fused_swiglu_scale(x, scale)
        # SiLU(100) ~ 100, SiLU(-100) ~ 0; verify no NaN/Inf
        out_np = out.numpy()
        self.assertTrue(
            np.all(np.isfinite(out_np)),
            f"Output contains non-finite values: {out_np}",
        )
        paddle.enable_static()

    def test_near_zero(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.full((2, 8), 1e-7, dtype='float32'))
        scale = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
        out = paddle._C_ops.fused_swiglu_scale(x, scale)
        self.assertTrue(np.all(np.isfinite(out.numpy())))
        paddle.enable_static()


class TestFusedSwigluScaleStress(unittest.TestCase):
    """Test determinism and memory stability under repeated calls."""

    def test_repeated_calls_deterministic(self):
        paddle.disable_static()
        x = paddle.randn([8, 4096], dtype='float32')
        scale = paddle.to_tensor(
            np.random.uniform(0.5, 1.5, [2048]).astype('float32')
        )

        results = []
        for _ in range(5):
            out = paddle._C_ops.fused_swiglu_scale(x, scale)
            results.append(out.numpy().copy())

        # Verify deterministic output
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0],
                results[i],
                err_msg="Repeated calls produced different outputs",
            )
        paddle.enable_static()


support_types = get_xpu_op_support_types('fused_swiglu_scale')
for stype in support_types:
    create_test_class(globals(), XPUTestFusedSwigluScaleOp, stype)

if __name__ == '__main__':
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()
