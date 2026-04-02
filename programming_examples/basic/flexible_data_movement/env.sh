#!/bin/bash
# Environment setup for flexible_data_movement prototypes
# Usage: source env.sh

source /scratch/jmelber/mlir-aie/ironenv/bin/activate

# Add pyxrt and XRT tools
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH

# Set PEANO and MLIR_AIE from pip-installed wheels
export MLIR_AIE_DIR=$(python3 -c "from aie.utils.config import root_path; print(root_path())" 2>/dev/null)
export PEANO_INSTALL_DIR=$(python3 -c "from importlib.metadata import distribution; import os; d = distribution('llvm-aie'); print(os.path.join(d._path.parent, 'llvm-aie'))" 2>/dev/null)
export PATH=${MLIR_AIE_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${MLIR_AIE_DIR}/lib:$LD_LIBRARY_PATH

# Auto-detect NPU2 (Strix/Strix Halo/Krackan)
NPU_INFO=$(/opt/xilinx/xrt/bin/xrt-smi examine 2>/dev/null | tr -d '\r')
if echo "$NPU_INFO" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
    export NPU2=1
    export DEVICE=npu2
else
    export NPU2=0
    export DEVICE=npu
fi

echo "MLIR_AIE_DIR:      $MLIR_AIE_DIR"
echo "PEANO_INSTALL_DIR: $PEANO_INSTALL_DIR"
echo "DEVICE:            $DEVICE (NPU2=$NPU2)"
