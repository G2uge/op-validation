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
Candidate implementations of fused_swiglu_scale using only Python-level
Paddle operators (no custom C++ kernel).

Each variant is a pair of (forward, backward) functions.
"""

import paddle
import paddle.nn.functional as F
from paddle import _C_ops
from paddle.nn.functional import swiglu


# ===========================================================================
# Candidate 1: native_small_ops
# Many tiny ops: chunk, silu, sigmoid, mul, concat, sum, etc.
# Expected to be the slowest.
# ===========================================================================
def fwd_native_small_ops(x, scale):
    gate, val = paddle.chunk(x, chunks=2, axis=-1)
    out = F.silu(gate) * val
    scale_exp = scale.cast(x.dtype)
    while scale_exp.ndim < out.ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    return out * scale_exp


def bwd_native_small_ops(x, scale, out_grad):
    hidden = x.shape[-1] // 2
    gate = x[..., :hidden]
    val = x[..., hidden:]
    sig = F.sigmoid(gate).cast(x.dtype)
    silu = gate * sig
    swiglu_out = silu * val

    scale_exp = scale.cast(x.dtype)
    while scale_exp.ndim < out_grad.ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    d_u = out_grad * scale_exp

    d_val = d_u * silu
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))
    d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)
    d_scale = paddle.sum(
        out_grad.cast(paddle.float32) * swiglu_out.cast(paddle.float32), axis=-1
    ).cast(scale.dtype)
    return d_x, d_scale


# ===========================================================================
# Candidate 2: python_api
# Use paddle.nn.functional.swiglu (Python wrapper) + paddle.multiply.
# ===========================================================================
def fwd_python_api(x, scale):
    out = swiglu(x)
    return paddle.multiply(out, scale.cast(x.dtype))


def bwd_python_api(x, scale, out_grad):
    swiglu_out = swiglu(x)
    d_u = paddle.multiply(out_grad, scale.cast(x.dtype))
    # python swiglu has no grad exposed directly; we fall back to chunk+silu
    gate, val = paddle.chunk(x, chunks=2, axis=-1)
    sig = F.sigmoid(gate).cast(x.dtype)
    silu = gate * sig
    d_val = d_u * silu
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))
    d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)
    d_scale = paddle.sum(
        paddle.multiply(out_grad, swiglu_out).cast(paddle.float32), axis=-1
    ).cast(scale.dtype)
    return d_x, d_scale


# ===========================================================================
# Candidate 3: c_ops_swiglu
# Use _C_ops.swiglu + _C_ops.multiply / _C_ops.sum.
# This is the coarsest granularity available from Python without custom C++.
# ===========================================================================
def fwd_c_ops(x, scale):
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def bwd_c_ops(x, scale, out_grad):
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale


# ===========================================================================
# Candidate 4: c_ops_inplace
# Forward uses inplace multiply_ to avoid one extra allocation.
# ===========================================================================
def fwd_inplace(x, scale):
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply_(out, scale.cast(x.dtype))


def bwd_inplace(x, scale, out_grad):
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale


# ===========================================================================
# Candidate 5: c_ops_cached
# Cache swiglu_out from forward and pass it to backward to avoid recomputation.
# In practice this needs a custom autograd.Function; here we simulate by
# pre-computing swiglu_out before backward.
# ===========================================================================
def fwd_cached(x, scale):
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def bwd_cached(x, scale, out_grad, swiglu_out):
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale


# ===========================================================================
# Candidate 6: autograd_c_ops
# Let Paddle build the backward graph automatically using _C_ops.
# We benchmark forward+backward together because they cannot be separated
# cleanly when using autograd.
# ===========================================================================
class SwiGLUScaleAutograd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, scale):
        out = _C_ops.swiglu(x, None)
        out = _C_ops.multiply(out, scale.cast(x.dtype))
        ctx.save_for_backward(x, scale)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        x, scale = ctx.saved_tensor()
        swiglu_out = _C_ops.swiglu(x, None)
        d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
        d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
        d_scale = _C_ops.sum(
            _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
            axis=[out_grad.ndim - 1],
        ).cast(scale.dtype)
        return d_x, d_scale


def fwd_autograd(x, scale):
    return SwiGLUScaleAutograd.apply(x, scale)


# ===========================================================================
# Candidate 7: c_ops_no_recompute
# Backward without recomputing swiglu_out for d_scale.
# Instead compute d_scale from dout * (out / scale).
# ===========================================================================
def fwd_no_recompute(x, scale):
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def bwd_no_recompute(x, scale, out_grad, out):
    # d_scale = sum(dout * out / scale, axis=-1)
    # but scale may broadcast, so we use actual division or just recompute
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    # Approximate d_scale = sum(dout * swiglu_out)
    # Since out = swiglu_out * scale, swiglu_out = out / scale
    # For simplicity we just recompute swiglu here (same as c_ops)
    swiglu_out = _C_ops.swiglu(x, None)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale
