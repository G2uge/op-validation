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
Refactored fused swiglu + scale using coarse-grained Paddle operators.

Instead of manually decomposing swiglu into sigmoid / mul / concat etc.,
this version calls the native Paddle ``swiglu`` / ``swiglu_grad`` kernels
(`paddle._C_ops.swiglu` / `paddle._C_ops.swiglu_grad`) together with
``paddle.multiply`` and ``paddle.sum``.
"""

import paddle
from paddle import _C_ops


def fused_swiglu_scale_forward(x, scale):
    """
    Forward: out = swiglu(x) * scale

    On CUDA the fused custom op is used; otherwise we fall back to the
    coarse-grained Paddle ops.
    """
    if paddle.is_compiled_with_cuda():
        # Custom fused kernel (only available when CUDA build includes it)
        from paddlefleet.ops import fused_swiglu_scale
        return fused_swiglu_scale(x, scale)

    # 1. swiglu(x)  -- last dim is split into gate / value automatically
    out = _C_ops.swiglu(x, None)

    # 2. multiply with scale (Paddle multiply supports broadcast natively)
    return paddle.multiply(out, scale.cast(x.dtype))


def fused_swiglu_scale_backward(x, scale, out_grad):
    """
    Backward of fused_swiglu_scale_forward.

    Returns (d_x, d_scale).
    """
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale_bwd
        return fused_swiglu_scale_bwd(x, scale, out_grad)

    # 1. Recompute swiglu output for d_scale
    swiglu_out = _C_ops.swiglu(x, None)

    # 2. d_u = dout * scale (broadcast handled by multiply)
    d_u = paddle.multiply(out_grad, scale.cast(x.dtype))

    # 3. swiglu_grad: dx from x and d_u
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)

    # 4. d_scale = sum(dout * swiglu_out, axis=-1)
    d_scale = paddle.sum(
        paddle.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=-1,
    ).cast(scale.dtype)

    return d_x, d_scale
