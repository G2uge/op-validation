#!/bin/bash
set -e

# 定位到脚本所在目录（xprofiling 文件夹）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ==================== 路径配置 ====================
PYTHON_ENV_PATH="/root/paddlejob/Gruge/envs/py310_paddleFormers"
XPROFILER_DIR="/root/paddlejob/Gruge/tools/xprofiler-Linux_x86_64-2.0.2.0"
XPROFILER_BIN="${XPROFILER_DIR}/bin/xprofiler"

OUTPUT_DIR="${SCRIPT_DIR}"
PROFILE_DIR="${OUTPUT_DIR}/profiles"
PROFILE_PREFIX="${PROFILE_DIR}/fused_swiglu_scale_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${PROFILE_DIR}"

# ==================== 基础环境 ====================
export LD_LIBRARY_PATH=${PYTHON_ENV_PATH}/lib/python3.10/site-packages/paddle/libs/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH="${XPROFILER_DIR}/so:${LD_LIBRARY_PATH}"

source "${PYTHON_ENV_PATH}/bin/activate"

# 安装 pybind_xprofiler（如未安装）
if ! python -c "import pybind_xprofiler" 2>/dev/null; then
    echo "[INFO] Installing pybind_xprofiler..."
    pip install "${XPROFILER_DIR}/python/pybind_xprofiler-2.0.2.0-py3-none-any.whl"
fi

# ==================== 启动 xprofiler Daemon ====================
echo "[INFO] Starting xprofiler daemon..."
nohup "${XPROFILER_BIN}" -r 500 --xpu=0 -e "${PROFILE_PREFIX}" -d \
    > "${OUTPUT_DIR}/xprofiler_daemon.log" 2>&1 &

XPID=$!
echo "[INFO] xprofiler daemon PID: $XPID"

# 等待 .sock 文件生成（最多 30 秒）
SOCK_FILE="${PWD}/xprofiler.sock"
WAIT_COUNT=0
while [ ! -S "${SOCK_FILE}" ] && [ $WAIT_COUNT -lt 30 ]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ ! -S "${SOCK_FILE}" ]; then
    echo "[ERROR] xprofiler .sock not generated after 30s!"
    cat "${OUTPUT_DIR}/xprofiler_daemon.log"
    kill -TERM $XPID 2>/dev/null || true
    exit 1
fi
echo "[INFO] xprofiler .sock ready: ${SOCK_FILE}"

# ==================== Client 环境变量 ====================
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_TRACING_OUTPUT_NAME="${SOCK_FILE}"
export NVTX_INJECTION64_PATH="${XPROFILER_DIR}/so/libxpuToolsExt.so"
export XPUTX_RUN_MODE=client
export XPUTX_LISTEN_ADDR="unix:${PWD}/xputx.sock"

# ==================== 运行 Profile ====================
echo "[INFO] Running profile script..."
python "${SCRIPT_DIR}/profile_fused_swiglu_scale.py"

# ==================== 清理 ====================
echo "[INFO] Stopping xprofiler daemon..."
kill -TERM $XPID 2>/dev/null || true
sleep 2
if ps -p $XPID > /dev/null 2>&1; then
    kill -KILL $XPID 2>/dev/null || true
fi

# ==================== 结果检查 ====================
echo ""
echo "=========================================="
OUTPUT_FILES=$(ls "${PROFILE_PREFIX}"* 2>/dev/null || true)
if [ -n "${OUTPUT_FILES}" ]; then
    echo "Profile data saved successfully!"
    ls -lh "${PROFILE_PREFIX}"*
    echo ""
    echo "View method:"
    echo "  1. Download trace files to local machine"
    echo "  2. Open Chrome browser at chrome://tracing/"
    echo "  3. Load the trace file"
else
    echo "[WARNING] No profile output files found!"
    echo "Check daemon log: ${OUTPUT_DIR}/xprofiler_daemon.log"
fi
echo "=========================================="
