#!/usr/bin/env python3
"""
Benchmark the native (non-fused) implementation of swiglu + scale,
corresponding to the else-branch in:
    /root/paddlejob/tmp/repos/PaddleFleet/src/paddlefleet/fusions/fused_swiglu_scale.py

Measures forward and backward latency with warm-up + averaging.
"""

import json
import time
import sys

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")
sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/test/legacy_test")

import paddle
import paddle.nn.functional as F
from paddle.nn.functional import swiglu


def fused_swiglu_scale_forward_native(x, scale):
    """Native forward (else-branch)."""
    out = swiglu(x)
    scale_exp = scale.cast(x.dtype)
    while scale_exp.ndim < out.ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    return out * scale_exp


def fused_swiglu_scale_backward_native(x, scale, out_grad):
    """Native backward (else-branch)."""
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


def benchmark(name, x_shape, scale_shape, warmup=10, iters=100):
    x = paddle.randn(x_shape, dtype=paddle.float32)
    scale = paddle.randn(scale_shape, dtype=paddle.float32)
    x.stop_gradient = False

    # warm-up forward
    for _ in range(warmup):
        out = fused_swiglu_scale_forward_native(x, scale)
    paddle.device.synchronize()

    # benchmark forward
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fused_swiglu_scale_forward_native(x, scale)
    paddle.device.synchronize()
    t1 = time.perf_counter()
    fwd_ms = (t1 - t0) * 1000.0 / iters

    # prepare backward
    out_grad = paddle.randn(out.shape, dtype=paddle.float32)

    # warm-up backward
    for _ in range(warmup):
        d_x, d_scale = fused_swiglu_scale_backward_native(x, scale, out_grad)
    paddle.device.synchronize()

    # benchmark backward
    t0 = time.perf_counter()
    for _ in range(iters):
        d_x, d_scale = fused_swiglu_scale_backward_native(x, scale, out_grad)
    paddle.device.synchronize()
    t1 = time.perf_counter()
    bwd_ms = (t1 - t0) * 1000.0 / iters

    return {
        "name": name,
        "x_shape": "x".join(str(d) for d in x_shape),
        "scale_shape": "x".join(str(d) for d in scale_shape),
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
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
    for name, x_shape, scale_shape in configs:
        print(f"Benchmarking {name} ...", file=sys.stderr)
        results.append(benchmark(name, x_shape, scale_shape))

    print(json.dumps({"benchmarks": results}, indent=2))


if __name__ == "__main__":
    main()
