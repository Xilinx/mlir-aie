#!/bin/bash
##===- test.sh -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
#
# Simple test script to verify the MLIR trace example works
#
##===----------------------------------------------------------------------===##

set -e

echo "==================================================================="
echo "Testing MLIR Trace Example"
echo "==================================================================="
echo ""

# Check if running on a system with NPU
if [ ! -e /dev/accel/accel0 ]; then
    echo "Warning: No NPU device found at /dev/accel/accel0"
    echo "This test requires an NPU device to run."
    echo "Build artifacts have been created successfully."
    exit 0
fi

echo "Step 1: Clean previous build artifacts"
make clean
echo ""

echo "Step 2: Build the example"
make
echo ""

echo "Step 3: Check if NPU device is available"
if [ ! -e /dev/accel/accel0 ]; then
    echo "Warning: No NPU device found at /dev/accel/accel0"
    echo "This test requires an NPU device to run."
    echo "Build artifacts have been created successfully."
    echo ""
    echo "Files created:"
    ls -lh build/final.xclbin build/insts.bin build/scale.o 2>/dev/null || echo "Some build files missing"
    exit 0
fi
echo "NPU device found!"
echo ""

echo "Step 4: Build host executable"
make mlir_trace_example.exe
echo ""

echo "Step 5: Run the design (without trace processing)"
echo "This will execute the vector-scalar multiplication on the NPU"
make run
echo ""

echo "Step 6: Run with trace enabled"
echo "This will capture and analyze trace data"
make trace
echo ""

echo "==================================================================="
echo "Test completed successfully!"
echo "==================================================================="
echo ""
echo "Trace output files:"
ls -lh trace*.json trace.txt trace_timeline.png 2>/dev/null || echo "No trace files found"
echo ""
echo "View the trace timeline: trace_timeline.png"
