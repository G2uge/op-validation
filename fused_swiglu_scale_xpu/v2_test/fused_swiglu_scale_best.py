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
Best Python-level implementation of fused_swiglu_scale for XPU.

Benchmark result (see benchmark_candidates.py):
- native_small_ops : total ~0.52 ms  (small_2D)
- python_api       : total ~0.42 ms
- autograd_c_ops   : total ~0.29 ms
- c_ops_cached     : total ~0.19 ms
- c_ops_swiglu     : total ~0.18 ms  <-- WINNER
- c_ops_inplace    : total ~0.18 ms  (same as winner, but inplace is less safe)

Winner: c_ops_swiglu -- uses _C_ops.swiglu + _C_ops.multiply + _C_ops.swiglu_grad.
This is the coarsest-grained composition available from Python without
writing any custom C++ kernel.
"""

import paddle
from paddle import _C_ops


def fused_swiglu_scale_forward(x, scale):
    """
    Forward: out = swiglu(x) * scale

    On CUDA the fused custom op is used; otherwise we fall back to the
    coarsest-grained Paddle ops available from Python.
    """
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale
        return fused_swiglu_scale(x, scale)

    # XPU / CPU fallback: _C_ops.swiglu (XDNN kernel) + _C_ops.multiply
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def fused_swiglu_scale_backward(x, scale, out_grad):
    """
    Backward of fused_swiglu_scale_forward.

    Returns (d_x, d_scale).
    """
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale_bwd
        return fused_swiglu_scale_bwd(x, scale, out_grad)

    # XPU / CPU fallback: manual grad with _C_ops.swiglu_grad
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale
