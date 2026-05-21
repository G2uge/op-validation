#!/usr/bin/env python3
"""
Benchmark best Python-level (c_ops_swiglu) vs native vs fused XPU kernel.

Uses the same benchmark methodology and data shapes as
benchmark_fused_swiglu_scale_paddle.py.

Three paths compared:
1. Native: chunk + silu + mul (standard ops) — from benchmark_fused_swiglu_scale_paddle.py
2. Best Python: _C_ops.swiglu + _C_ops.multiply + _C_ops.swiglu_grad
3. Fused XPU: paddle._C_ops.fused_swiglu_scale (reference)
"""

import json
import sys
import time

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")

import paddle
from paddle import _C_ops


# ---------------------------------------------------------------------------
# Native implementation (same as benchmark_fused_swiglu_scale_paddle.py)
# ---------------------------------------------------------------------------
def native_swiglu_scale(x, scale):
    """Native Paddle implementation using standard ops."""
    gate, val = paddle.chunk(x, chunks=2, axis=-1)
    out = paddle.nn.functional.silu(gate) * val * scale
    return out


def _broadcast_reduce(grad, target_shape):
    """Sum-reduce grad to match target_shape, handling broadcasting."""
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
    """Native backward by hand-written gradient."""
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
# Best Python-level implementation (c_ops_swiglu)
# ---------------------------------------------------------------------------
def best_python_swiglu_scale(x, scale):
    """Best Python-level: _C_ops.swiglu + _C_ops.multiply."""
    out = _C_ops.swiglu(x, None)
    return _C_ops.multiply(out, scale.cast(x.dtype))


def best_python_swiglu_scale_backward(x, scale, out_grad):
    """Best Python-level backward: _C_ops.swiglu_grad + _C_ops.multiply + _C_ops.sum."""
    swiglu_out = _C_ops.swiglu(x, None)
    d_u = _C_ops.multiply(out_grad, scale.cast(x.dtype))
    d_x, _ = _C_ops.swiglu_grad(x, None, d_u)
    d_scale = _C_ops.sum(
        _C_ops.multiply(out_grad, swiglu_out).cast(paddle.float32),
        axis=[out_grad.ndim - 1],
    ).cast(scale.dtype)
    return d_x, d_scale


# ---------------------------------------------------------------------------
# Fused XPU kernel (reference)
# ---------------------------------------------------------------------------
def fused_swiglu_scale(x, scale):
    """Fused XPU op from paddle._C_ops."""
    return paddle._C_ops.fused_swiglu_scale(x, scale)


def fused_swiglu_scale_backward(x, scale, out_grad):
    """Fused XPU backward via direct op call."""
    x_grad, scale_grad = paddle._C_ops.fused_swiglu_scale_grad(x, scale, out_grad)
    return x_grad, scale_grad


# ---------------------------------------------------------------------------
# Benchmark helper (same as benchmark_fused_swiglu_scale_paddle.py)
# ---------------------------------------------------------------------------
def benchmark_forward(impl_name, fn, x, scale, warmup=10, iters=100):
    for _ in range(warmup):
        out = fn(x, scale)
    paddle.device.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(x, scale)
    paddle.device.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def benchmark_backward(impl_name, fn_fwd, fn_bwd, x, scale, warmup=10, iters=100):
    # Forward to get output shape
    with paddle.no_grad():
        out = fn_fwd(x, scale)
    out_grad = paddle.randn(out.shape, dtype=x.dtype)

    for _ in range(warmup):
        fn_bwd(x, scale, out_grad)
    paddle.device.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn_bwd(x, scale, out_grad)
    paddle.device.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


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

    results = []
    for name, x_shape, scale_shape in configs:
        print(f"Benchmarking {name} ...", file=sys.stderr)

        x = paddle.randn(x_shape, dtype=paddle.float32)
        scale = paddle.randn(scale_shape, dtype=paddle.float32)

        # Native forward
        native_fwd_ms = benchmark_forward("native_fwd", native_swiglu_scale, x, scale)

        # Native backward
        native_bwd_ms = benchmark_backward(
            "native_bwd", native_swiglu_scale, native_swiglu_scale_backward, x, scale
        )

        # Best Python forward
        best_fwd_ms = benchmark_forward("best_fwd", best_python_swiglu_scale, x, scale)

        # Best Python backward
        best_bwd_ms = benchmark_backward(
            "best_bwd", best_python_swiglu_scale, best_python_swiglu_scale_backward, x, scale
        )

        # Fused forward
        fused_fwd_ms = benchmark_forward("fused_fwd", fused_swiglu_scale, x, scale)

        # Fused backward
        fused_bwd_ms = benchmark_backward(
            "fused_bwd", fused_swiglu_scale, fused_swiglu_scale_backward, x, scale
        )

        entry = {
            "name": name,
            "x_shape": "x".join(str(d) for d in x_shape),
            "scale_shape": "x".join(str(d) for d in scale_shape),
            "native_forward_ms": native_fwd_ms,
            "native_backward_ms": native_bwd_ms,
            "best_forward_ms": best_fwd_ms,
            "best_backward_ms": best_bwd_ms,
            "fused_forward_ms": fused_fwd_ms,
            "fused_backward_ms": fused_bwd_ms,
            "best_vs_native_fwd": native_fwd_ms / best_fwd_ms if best_fwd_ms > 0 else None,
            "best_vs_native_bwd": native_bwd_ms / best_bwd_ms if best_bwd_ms > 0 else None,
            "fused_vs_native_fwd": native_fwd_ms / fused_fwd_ms if fused_fwd_ms > 0 else None,
            "fused_vs_native_bwd": native_bwd_ms / fused_bwd_ms if fused_bwd_ms > 0 else None,
            "fused_vs_best_fwd": best_fwd_ms / fused_fwd_ms if fused_fwd_ms > 0 else None,
            "fused_vs_best_bwd": best_bwd_ms / fused_bwd_ms if fused_bwd_ms > 0 else None,
        }

        results.append(entry)

    print(json.dumps({"benchmarks": results}, indent=2))

    # -----------------------------------------------------------------------
    # Pretty summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 130, file=sys.stderr)
    print(
        f"{'Config':<15} {'Native Fwd':>12} {'Best Fwd':>12} {'Fused Fwd':>12} "
        f"{'Native Bwd':>12} {'Best Bwd':>12} {'Fused Bwd':>12}",
        file=sys.stderr,
    )
    print("=" * 130, file=sys.stderr)
    for r in results:
        print(
            f"{r['name']:<15} "
            f"{r['native_forward_ms']:>11.3f}ms "
            f"{r['best_forward_ms']:>11.3f}ms "
            f"{r['fused_forward_ms']:>11.3f}ms "
            f"{r['native_backward_ms']:>11.3f}ms "
            f"{r['best_backward_ms']:>11.3f}ms "
            f"{r['fused_backward_ms']:>11.3f}ms",
            file=sys.stderr,
        )
        print(
            f"{'':<15} "
            f"{'(base)':>12} "
            f"{r['best_vs_native_fwd']:>10.2f}x "
            f"{r['fused_vs_native_fwd']:>10.2f}x "
            f"{'(base)':>12} "
            f"{r['best_vs_native_bwd']:>10.2f}x "
            f"{r['fused_vs_native_bwd']:>10.2f}x",
            file=sys.stderr,
        )
        print("-" * 130, file=sys.stderr)

    print("\nSpeedup: best Python vs native, fused vs native, fused vs best Python", file=sys.stderr)
    print("=" * 130, file=sys.stderr)


if __name__ == "__main__":
    main()
