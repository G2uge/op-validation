#!/usr/bin/env python3
"""
Profile fused_swiglu_scale (forward + backward) with xprofiler 2.0.2.0 Daemon Mode.
Run under: /root/paddlejob/Gruge/envs/py310_paddleFormers
"""

import sys
import time

sys.path.insert(0, "/root/paddlejob/tmp/repos/Paddle/build/python")

import paddle

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
# Fused ops
# ---------------------------------------------------------------------------
def fused_swiglu_scale(x, scale):
    return paddle._C_ops.fused_swiglu_scale(x, scale)

def fused_swiglu_scale_backward(x, scale, out_grad):
    x_grad, scale_grad = paddle._C_ops.fused_swiglu_scale_grad(x, scale, out_grad)
    return x_grad, scale_grad

# ---------------------------------------------------------------------------
# Profile helper
# ---------------------------------------------------------------------------
def profile_case(name, x_shape, scale_shape, warmup=10, iters=20):
    print(f"[CASE] {name:15s}  x={str(x_shape):20s}  scale={str(scale_shape)}", file=sys.stderr)

    x = paddle.randn(x_shape, dtype=paddle.float32)
    scale = paddle.randn(scale_shape, dtype=paddle.float32)

    # ---- warmup forward ----
    for _ in range(warmup):
        out = fused_swiglu_scale(x, scale)
    paddle.device.synchronize()

    # ---- profile forward ----
    for _ in range(iters):
        out = fused_swiglu_scale(x, scale)
    paddle.device.synchronize()

    # ---- prepare & warmup backward ----
    out_grad = paddle.randn(out.shape, dtype=x.dtype)
    for _ in range(warmup):
        fused_swiglu_scale_backward(x, scale, out_grad)
    paddle.device.synchronize()

    # ---- profile backward ----
    for _ in range(iters):
        fused_swiglu_scale_backward(x, scale, out_grad)
    paddle.device.synchronize()

    # timeline gap for easy separation in chrome tracing
    time.sleep(0.3)
    paddle.device.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    paddle.set_device("xpu")

    configs = [
        ("small_2D",  [4, 8],          [4, 1]),
        ("medium_2D", [1024, 256],     [1024, 1]),
        ("large_2D",  [4096, 512],     [4096, 1]),
        ("small_3D",  [2, 3, 16],      [1, 1, 8]),
        ("medium_3D", [8, 128, 512],   [8, 128, 256]),
        ("large_3D",  [32, 512, 1024], [32, 512, 512]),
    ]

    if not HAS_XPROFILER:
        print("[ERROR] pybind_xprofiler is required for profiling.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] xprofiler start", file=sys.stderr)
    pybind_xprofiler.cuda_profiler_start()

    try:
        for name, x_shape, scale_shape in configs:
            profile_case(name, x_shape, scale_shape)
    finally:
        pybind_xprofiler.cuda_profiler_stop()
        print("[INFO] xprofiler stop", file=sys.stderr)

    print("[INFO] All cases finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
