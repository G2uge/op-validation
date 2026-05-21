#!/usr/bin/env python3
"""
Benchmark all Python-level fused_swiglu_scale candidates on XPU.

Candidates:
1. native_small_ops   – many tiny ops (chunk, silu, sigmoid, mul, concat, sum)
2. python_api         – paddle.nn.functional.swiglu + paddle.multiply
3. c_ops_swiglu       – _C_ops.swiglu + _C_ops.multiply + _C_ops.swiglu_grad
4. c_ops_inplace      – same as 3 but forward uses multiply_ (inplace)
5. c_ops_cached       – backward caches swiglu_out to avoid recomputation
6. autograd_c_ops     – PyLayer autograd with _C_ops

The winner will be the coarsest-grained Python composition (c_ops_swiglu).
"""

import json
import sys
import time

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")
sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/test/legacy_test")

import paddle
from paddle import _C_ops

from fused_swiglu_scale_candidates import (
    bwd_cached,
    bwd_c_ops,
    bwd_inplace,
    bwd_native_small_ops,
    bwd_python_api,
    fwd_cached,
    fwd_c_ops,
    fwd_inplace,
    fwd_native_small_ops,
    fwd_no_recompute,
    fwd_python_api,
    fwd_autograd,
    SwiGLUScaleAutograd,
)

WARMUP = 10
ITERS = 100


def benchmark_manual(name, fwd_fn, bwd_fn, x_shape, scale_shape, use_cached=False):
    x = paddle.randn(x_shape, dtype=paddle.float32)
    scale = paddle.randn(scale_shape, dtype=paddle.float32)

    # Forward warm-up
    for _ in range(WARMUP):
        out = fwd_fn(x, scale)
    paddle.device.synchronize()

    # Forward benchmark
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fwd_fn(x, scale)
    paddle.device.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

    out_grad = paddle.randn(out.shape, dtype=paddle.float32)

    # Backward warm-up
    for _ in range(WARMUP):
        if use_cached:
            with paddle.no_grad():
                swiglu_out = _C_ops.swiglu(x, None)
            d_x, d_scale = bwd_fn(x, scale, out_grad, swiglu_out)
        else:
            d_x, d_scale = bwd_fn(x, scale, out_grad)
    paddle.device.synchronize()

    # Backward benchmark
    t0 = time.perf_counter()
    for _ in range(ITERS):
        if use_cached:
            with paddle.no_grad():
                swiglu_out = _C_ops.swiglu(x, None)
            d_x, d_scale = bwd_fn(x, scale, out_grad, swiglu_out)
        else:
            d_x, d_scale = bwd_fn(x, scale, out_grad)
    paddle.device.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

    return {
        "name": name,
        "x_shape": "x".join(str(d) for d in x_shape),
        "scale_shape": "x".join(str(d) for d in scale_shape),
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
    }


def benchmark_autograd(name, x_shape, scale_shape):
    x = paddle.randn(x_shape, dtype=paddle.float32)
    scale = paddle.randn(scale_shape, dtype=paddle.float32)

    # Warm-up forward+backward
    for _ in range(WARMUP):
        x_tmp = x.detach()
        x_tmp.stop_gradient = False
        scale_tmp = scale.detach()
        scale_tmp.stop_gradient = False
        out = SwiGLUScaleAutograd.apply(x_tmp, scale_tmp)
        out_grad = paddle.randn(out.shape, dtype=paddle.float32)
        paddle.autograd.backward([out], [out_grad])
    paddle.device.synchronize()

    # Benchmark forward+backward together
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x_tmp = x.detach()
        x_tmp.stop_gradient = False
        scale_tmp = scale.detach()
        scale_tmp.stop_gradient = False
        out = SwiGLUScaleAutograd.apply(x_tmp, scale_tmp)
        out_grad = paddle.randn(out.shape, dtype=paddle.float32)
        paddle.autograd.backward([out], [out_grad])
    paddle.device.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

    return {
        "name": name,
        "x_shape": "x".join(str(d) for d in x_shape),
        "scale_shape": "x".join(str(d) for d in scale_shape),
        "forward_ms": None,
        "backward_ms": total_ms,
    }


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
    for cfg_name, x_shape, scale_shape in configs:
        print(f"\nConfig: {cfg_name} x={x_shape} scale={scale_shape}", file=sys.stderr)

        # 1. native_small_ops
        print("  [native_small_ops] ...", file=sys.stderr)
        results.append(benchmark_manual("native_small_ops", fwd_native_small_ops, bwd_native_small_ops, x_shape, scale_shape))

        # 2. python_api
        print("  [python_api] ...", file=sys.stderr)
        results.append(benchmark_manual("python_api", fwd_python_api, bwd_python_api, x_shape, scale_shape))

        # 3. c_ops_swiglu
        print("  [c_ops_swiglu] ...", file=sys.stderr)
        results.append(benchmark_manual("c_ops_swiglu", fwd_c_ops, bwd_c_ops, x_shape, scale_shape))

        # 4. c_ops_inplace
        print("  [c_ops_inplace] ...", file=sys.stderr)
        results.append(benchmark_manual("c_ops_inplace", fwd_inplace, bwd_inplace, x_shape, scale_shape))

        # 5. c_ops_cached
        print("  [c_ops_cached] ...", file=sys.stderr)
        results.append(benchmark_manual("c_ops_cached", fwd_cached, bwd_cached, x_shape, scale_shape, use_cached=True))

        # 6. autograd_c_ops
        print("  [autograd_c_ops] ...", file=sys.stderr)
        results.append(benchmark_autograd("autograd_c_ops", x_shape, scale_shape))

    print(json.dumps({"benchmarks": results}, indent=2))

    # Pretty summary table
    print("\n" + "=" * 110, file=sys.stderr)
    print(
        f"{'Config':<15} {'Candidate':<18} {'Forward (ms)':>14} {'Backward (ms)':>14} {'Total (ms)':>14}",
        file=sys.stderr,
    )
    print("=" * 110, file=sys.stderr)

    for cfg_name, _, _ in configs:
        cfg_results = [r for r in results if r["x_shape"] == "x".join(str(d) for d in [r["x_shape"].split("x")])]
        # Actually iterate over results directly
        pass

    # Simpler: just iterate results
    for r in results:
        fwd = f"{r['forward_ms']:.4f}" if r["forward_ms"] is not None else "N/A"
        bwd = f"{r['backward_ms']:.4f}" if r["backward_ms"] is not None else "N/A"
        total = ""
        if r["forward_ms"] is not None and r["backward_ms"] is not None:
            total = f"{r['forward_ms'] + r['backward_ms']:.4f}"
        elif r["backward_ms"] is not None:
            total = f"{r['backward_ms']:.4f} (total)"
        print(
            f"{r['name']:<15} {r.get('variant', ''):<18} {fwd:>14} {bwd:>14} {total:>14}",
            file=sys.stderr,
        )

    print("=" * 110, file=sys.stderr)


if __name__ == "__main__":
    main()
