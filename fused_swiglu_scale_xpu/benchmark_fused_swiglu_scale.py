#!/usr/bin/env python3
"""
Benchmark fused_swiglu_scale_forward and fused_swiglu_scale_backward
from /root/paddlejob/tmp/repos/PaddleFleet/src/paddlefleet/fusions/fused_swiglu_scale.py

Test cases mirror benchmark_fused_swiglu_scale_xpu.cc.
"""

import importlib.util
import json
import sys
import time

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")
sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/test/legacy_test")

import paddle

# Directly load the target module without triggering the full paddlefleet package.
_mod_path = "/root/paddlejob/tmp/repos/PaddleFleet/src/paddlefleet/fusions/fused_swiglu_scale.py"
_spec = importlib.util.spec_from_file_location("fused_swiglu_scale", _mod_path)
_fused_mod = importlib.util.module_from_spec(_spec)
_sys_modules_bak = sys.modules.get("paddlefleet")
try:
    sys.modules["paddlefleet"] = type(sys)("paddlefleet")  # dummy package
    _spec.loader.exec_module(_fused_mod)
finally:
    if _sys_modules_bak is None:
        sys.modules.pop("paddlefleet", None)
    else:
        sys.modules["paddlefleet"] = _sys_modules_bak

fused_swiglu_scale_forward = _fused_mod.fused_swiglu_scale_forward
fused_swiglu_scale_backward = _fused_mod.fused_swiglu_scale_backward


def benchmark(name, x_shape, scale_shape, warmup=10, iters=100):
    x = paddle.randn(x_shape, dtype=paddle.float32)
    scale = paddle.randn(scale_shape, dtype=paddle.float32)

    # warm-up forward
    for _ in range(warmup):
        out = fused_swiglu_scale_forward(x, scale)
    paddle.device.synchronize()

    # benchmark forward
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fused_swiglu_scale_forward(x, scale)
    paddle.device.synchronize()
    t1 = time.perf_counter()
    fwd_ms = (t1 - t0) * 1000.0 / iters

    # prepare backward
    out_grad = paddle.randn(out.shape, dtype=paddle.float32)

    # warm-up backward
    for _ in range(warmup):
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
    paddle.device.synchronize()

    # benchmark backward
    t0 = time.perf_counter()
    for _ in range(iters):
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
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
