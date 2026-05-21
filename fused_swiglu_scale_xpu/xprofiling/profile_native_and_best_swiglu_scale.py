#!/usr/bin/env python3
"""
Profile native vs best_python swiglu_scale with xprofiler 2.0.2.0 Daemon Mode.

Captures:
1. Native: chunk + silu + mul (multiple discrete kernels)
2. Best Python: _C_ops.swiglu + _C_ops.multiply + _C_ops.swiglu_grad + _C_ops.sum

Timeline layout per case:
  [native fwd] [native bwd] --gap-- [best fwd] [best bwd] --gap-- (next case)
"""

import sys
import time

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")

import paddle
from paddle import _C_ops

# ---------------------------------------------------------------------------
# xprofiler marker
# ---------------------------------------------------------------------------
try:
    import pybind_xprofiler
    HAS_XPROFILER = True
except ImportError:
    HAS_XPROFILER = False
    print("WARNING: pybind_xprofiler not found, running without profiler!", file=sys.stderr)

# ---------------------------------------------------------------------------
# Native implementation
# ---------------------------------------------------------------------------
def native_swiglu_scale(x, scale):
    gate, val = paddle.chunk(x, chunks=2, axis=-1)
    out = paddle.nn.functional.silu(gate) * val * scale
    return out


def _broadcast_reduce(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    grad_shape = list(grad.shape)
    padded_target = list(target_shape)
    while len(padded_target) < len(grad_shape):
        padded_target.insert(0, 1)
    reduce_dims = [i for i, (g, t) in enumerate(zip(grad_shape, padded_target)) if g != t]
    if reduce_dims:
        grad = paddle.sum(grad, axis=reduce_dims, keepdim=True)
    squeeze_dims = list(range(len(grad.shape) - len(target_shape)))
    if squeeze_dims:
        grad = paddle.squeeze(grad, axis=squeeze_dims)
    return grad


def native_swiglu_scale_backward(x, scale, out_grad):
    gate, val = paddle.chunk(x, chunks=2, axis=-1)
    silu_gate = paddle.nn.functional.silu(gate)
    sigmoid_gate = paddle.nn.functional.sigmoid(gate)
    silu_prime = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    gate_grad = out_grad * silu_prime * val * scale
    val_grad = out_grad * silu_gate * scale
    x_grad = paddle.concat([gate_grad, val_grad], axis=-1)
    scale_grad = _broadcast_reduce(out_grad * silu_gate * val, scale.shape)
    return x_grad, scale_grad


# ---------------------------------------------------------------------------
# Best Python-level implementation
# ---------------------------------------------------------------------------
def best_python_swiglu_scale(x, scale):
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def best_python_swiglu_scale_backward(x, scale, out_grad):
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def _profile_loop(fn, x, scale, warmup=10, iters=20):
    for _ in range(warmup):
        fn(x, scale)
    paddle.device.synchronize()
    for _ in range(iters):
        fn(x, scale)
    paddle.device.synchronize()


def _profile_loop_bwd(fn, x, scale, out_grad, warmup=10, iters=20):
    for _ in range(warmup):
        fn(x, scale, out_grad)
    paddle.device.synchronize()
    for _ in range(iters):
        fn(x, scale, out_grad)
    paddle.device.synchronize()


def profile_native(x, scale, out_grad, warmup=10, iters=20):
    """Profile native forward + backward."""
    _profile_loop(native_swiglu_scale, x, scale, warmup, iters)
    _profile_loop_bwd(native_swiglu_scale_backward, x, scale, out_grad, warmup, iters)


def profile_best(x, scale, out_grad, warmup=10, iters=20):
    """Profile best_python forward + backward."""
    _profile_loop(best_python_swiglu_scale, x, scale, warmup, iters)
    _profile_loop_bwd(best_python_swiglu_scale_backward, x, scale, out_grad, warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    paddle.set_device("xpu")

    configs = [
        ("small_2D", [4, 8], [4, 1]),
        ("medium_2D", [1024, 256], [1024, 1]),
        ("large_2D", [4096, 512], [4096, 1]),
        ("small_3D", [2, 3, 16], [1, 1, 8]),
        ("medium_3D", [8, 128, 512], [8, 128, 256]),
        ("large_3D", [32, 512, 1024], [32, 512, 512]),
    ]

    if not HAS_XPROFILER:
        print("[ERROR] pybind_xprofiler is required for profiling.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] xprofiler start", file=sys.stderr)
    pybind_xprofiler.cuda_profiler_start()

    try:
        for name, x_shape, scale_shape in configs:
            print(f"[CASE] {name}: x={x_shape}, scale={scale_shape}", file=sys.stderr)

            x = paddle.randn(x_shape, dtype=paddle.float32)
            scale = paddle.randn(scale_shape, dtype=paddle.float32)

            # Prepare shared out_grad (shape from native forward)
            with paddle.no_grad():
                out = native_swiglu_scale(x, scale)
            out_grad = paddle.randn(out.shape, dtype=x.dtype)
            paddle.device.synchronize()

            # ---- Native profile ----
            print(f"  [NATIVE]", file=sys.stderr)
            profile_native(x, scale, out_grad)

            # Gap: native vs best
            time.sleep(0.3)
            paddle.device.synchronize()

            # ---- Best profile ----
            print(f"  [BEST]", file=sys.stderr)
            profile_best(x, scale, out_grad)

            # Gap: next case
            time.sleep(0.5)
            paddle.device.synchronize()
    finally:
        pybind_xprofiler.cuda_profiler_stop()
        print("[INFO] xprofiler stop", file=sys.stderr)

    print("[INFO] All cases finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
