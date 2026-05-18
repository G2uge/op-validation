# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
Standalone numerical accuracy test for XPU swiglu_grad.

This file validates that:
    dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
produces numerically equivalent results to a pure-Paddle reference
implementation of the SwiGLU backward.

Run on an XPU machine:
    python test_swiglu_grad_xpu_accuracy.py -v

If XPU is not compiled in, all tests are skipped automatically.
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F


def _reference_swiglu_grad(y, g):
    """Pure Paddle reference implementation of SwiGLU backward.

    SwiGLU forward splits ``y`` into two halves along the last dimension:
        y1, y2 = split(y, 2, axis=-1)
        out = silu(y1) * y2

    Given gradient ``g`` (same shape as ``out``), the backward is:
        grad_y1 = g * y2 * sigmoid(y1) * (1 + y1 * (1 - sigmoid(y1)))
        grad_y2 = g * silu(y1)
        dx = concat([grad_y1, grad_y2], axis=-1)

    Args:
        y (paddle.Tensor): Input tensor of shape [..., D], where D is even.
        g (paddle.Tensor): Gradient of loss w.r.t. output, shape [..., D/2].

    Returns:
        paddle.Tensor: Gradient w.r.t. ``y``, shape [..., D].
    """
    y1, y2 = paddle.split(y, 2, axis=-1)
    sigmoid_y1 = F.sigmoid(y1)
    silu_y1 = y1 * sigmoid_y1

    grad_y1 = g * y2 * sigmoid_y1 * (1.0 + y1 * (1.0 - sigmoid_y1))
    grad_y2 = g * silu_y1

    return paddle.concat([grad_y1, grad_y2], axis=-1)


@unittest.skipIf(
    not paddle.is_compiled_with_xpu(),
    "XPU is not compiled in the current Paddle build. "
    "Please run this test on a machine with XPU available.",
)
class TestSwigluGradXPU(unittest.TestCase):
    """Numerical equivalence tests for XPU paddle._C_ops.swiglu_grad."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("xpu")

    def tearDown(self):
        paddle.set_device(self._orig_device)

    def _check_allclose(self, xpu_out, ref_out, rtol, atol, msg=""):
        xpu_np = xpu_out.numpy()
        ref_np = ref_out.numpy()
        diff = np.abs(xpu_np - ref_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        np.testing.assert_allclose(xpu_np, ref_np, rtol=rtol, atol=atol, err_msg=msg)
        print(f"  [{msg}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    def test_swiglu_grad_xpu_float32_2d(self):
        """2D float32 numerical equivalence."""
        y = paddle.randn([4, 8], dtype="float32")
        g = paddle.randn([4, 4], dtype="float32")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-5, atol=1e-5, msg="fp32_2d")

    def test_swiglu_grad_xpu_float16_2d(self):
        """2D float16 numerical equivalence."""
        y = paddle.randn([4, 8], dtype="float16")
        g = paddle.randn([4, 4], dtype="float16")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-3, atol=1e-3, msg="fp16_2d")

    def test_swiglu_grad_xpu_float32_3d(self):
        """3D float32 numerical equivalence (typical transformer MLP shape)."""
        y = paddle.randn([2, 3, 8], dtype="float32")
        g = paddle.randn([2, 3, 4], dtype="float32")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-5, atol=1e-5, msg="fp32_3d")

    def test_swiglu_grad_xpu_float16_3d(self):
        """3D float16 numerical equivalence."""
        y = paddle.randn([2, 3, 8], dtype="float16")
        g = paddle.randn([2, 3, 4], dtype="float16")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-3, atol=1e-3, msg="fp16_3d")

    def test_swiglu_grad_xpu_large_dim(self):
        """Large hidden dim float32 equivalence."""
        y = paddle.randn([8, 4096], dtype="float32")
        g = paddle.randn([8, 2048], dtype="float32")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-5, atol=1e-5, msg="fp32_large")

    def test_swiglu_grad_xpu_edge_broadcast(self):
        """Test with 1-element grad batch (broadcast-like scenario)."""
        y = paddle.randn([1, 8], dtype="float32")
        g = paddle.randn([1, 4], dtype="float32")

        xpu_dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
        ref_dx = _reference_swiglu_grad(y, g)

        self._check_allclose(xpu_dx, ref_dx, rtol=1e-5, atol=1e-5, msg="fp32_1batch")


if __name__ == "__main__":
    print(f"Paddle version: {paddle.__version__}")
    print(f"Compiled with XPU: {paddle.is_compiled_with_xpu()}")
    print(f"Current device: {paddle.get_device()}")
    print("-" * 60)
    unittest.main(verbosity=2)
