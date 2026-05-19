#!/usr/bin/env python3
"""
Run both C++ (fused XPU) and Python (native) benchmarks,
then print a side-by-side comparison table.
"""

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cpp_benchmark():
    cpp_bin = os.path.join(SCRIPT_DIR, "benchmark_fused_swiglu_scale_xpu")
    if not os.path.exists(cpp_bin):
        print(f"ERROR: {cpp_bin} not found. Please run ./build.sh first.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        "/root/paddlejob/tmp/repos/Paddle/build/python",
        "/root/paddlejob/tmp/repos/Paddle/test/legacy_test",
    ])

    result = subprocess.run(
        [cpp_bin],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        env=env,
    )
    if result.returncode != 0:
        print("C++ benchmark failed:\n" + result.stderr, file=sys.stderr)
        sys.exit(1)

    return json.loads(result.stdout)


def run_python_benchmark():
    py_script = os.path.join(SCRIPT_DIR, "benchmark_fused_swiglu_scale_native.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        "/root/paddlejob/tmp/repos/Paddle/build/python",
        "/root/paddlejob/tmp/repos/Paddle/test/legacy_test",
    ])

    result = subprocess.run(
        [sys.executable, py_script],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        env=env,
    )
    if result.returncode != 0:
        print("Python benchmark failed:\n" + result.stderr, file=sys.stderr)
        sys.exit(1)

    # The JSON is printed to stdout; filter out lines that look like warnings
    lines = [l for l in result.stdout.splitlines() if l.strip().startswith("{") or l.strip().startswith("[")]
    data = result.stdout[result.stdout.find("{"):]
    return json.loads(data)


def fmt_ms(v):
    return f"{v:.4f}"


def speedup(native, fused):
    if fused == 0:
        return "N/A"
    return f"{native / fused:.2f}x"


def main():
    print("=" * 100)
    print("Running C++ fused benchmark (XPU) ...")
    cpp_data = run_cpp_benchmark()
    print("Running Python native benchmark ...")
    py_data = run_python_benchmark()

    cpp_map = {b["name"]: b for b in cpp_data["benchmarks"]}
    py_map = {b["name"]: b for b in py_data["benchmarks"]}

    names = [b["name"] for b in cpp_data["benchmarks"]]

    print()
    print("=" * 100)
    print(f"{'Case':<18} {'Shape':<22} {'Metric':<10} {'Native (ms)':<14} {'Fused (ms)':<14} {'Speedup':<10}")
    print("-" * 100)

    total_fwd_native = 0.0
    total_fwd_fused = 0.0
    total_bwd_native = 0.0
    total_bwd_fused = 0.0

    for name in names:
        c = cpp_map[name]
        p = py_map[name]
        shape = f"{c['x_shape']} / {c['scale_shape']}"

        fwd_native = p["forward_ms"]
        fwd_fused = c["forward_ms"]
        bwd_native = p["backward_ms"]
        bwd_fused = c["backward_ms"]

        total_fwd_native += fwd_native
        total_fwd_fused += fwd_fused
        total_bwd_native += bwd_native
        total_bwd_fused += bwd_fused

        print(f"{name:<18} {shape:<22} {'forward':<10} {fmt_ms(fwd_native):<14} {fmt_ms(fwd_fused):<14} {speedup(fwd_native, fwd_fused):<10}")
        print(f"{'':<18} {'':<22} {'backward':<10} {fmt_ms(bwd_native):<14} {fmt_ms(bwd_fused):<14} {speedup(bwd_native, bwd_fused):<10}")

    print("-" * 100)
    print(f"{'AVERAGE':<18} {'':<22} {'forward':<10} {fmt_ms(total_fwd_native / len(names)):<14} {fmt_ms(total_fwd_fused / len(names)):<14} {speedup(total_fwd_native, total_fwd_fused):<10}")
    print(f"{'':<18} {'':<22} {'backward':<10} {fmt_ms(total_bwd_native / len(names)):<14} {fmt_ms(total_bwd_fused / len(names)):<14} {speedup(total_bwd_native, total_bwd_fused):<10}")
    print("=" * 100)

    # Save JSON report
    report = {
        "cases": [
            {
                "name": name,
                "x_shape": cpp_map[name]["x_shape"],
                "scale_shape": cpp_map[name]["scale_shape"],
                "forward_native_ms": py_map[name]["forward_ms"],
                "forward_fused_ms": cpp_map[name]["forward_ms"],
                "backward_native_ms": py_map[name]["backward_ms"],
                "backward_fused_ms": cpp_map[name]["backward_ms"],
            }
            for name in names
        ],
        "average": {
            "forward_native_ms": total_fwd_native / len(names),
            "forward_fused_ms": total_fwd_fused / len(names),
            "backward_native_ms": total_bwd_native / len(names),
            "backward_fused_ms": total_bwd_fused / len(names),
        },
    }
    report_path = os.path.join(SCRIPT_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
