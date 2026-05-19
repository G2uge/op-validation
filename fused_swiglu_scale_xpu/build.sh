#!/bin/bash
set -e

# Paddle build root
PADDLE_BUILD="/root/paddlejob/tmp/repos/Paddle/build"

# XPU headers
XPU_INCLUDE="${PADDLE_BUILD}/third_party/install/xpu/include"

# XPU libs
XPU_LIB="${PADDLE_BUILD}/third_party/install/xpu/lib"

# Compile accuracy test
g++ -std=c++11 \
  -I"${XPU_INCLUDE}" \
  test_fused_swiglu_scale_xpu.cc \
  -L"${XPU_LIB}" -lxpuapi -lxpurt -lcudart \
  -Wl,-rpath,"${XPU_LIB}" \
  -o test_fused_swiglu_scale_xpu

# Compile benchmark test
g++ -std=c++11 \
  -I"${XPU_INCLUDE}" \
  benchmark_fused_swiglu_scale_xpu.cc \
  -L"${XPU_LIB}" -lxpuapi -lxpurt -lcudart \
  -Wl,-rpath,"${XPU_LIB}" \
  -o benchmark_fused_swiglu_scale_xpu

echo "Build succeeded"
