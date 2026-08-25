# Copyright (C) 2018-2022 Xilinx, Inc.
# Copyright (C) 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Top-level CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(aie-pdi-transform)

# Set the C standard to C99
set(CMAKE_C_STANDARD 99)

# Add subdirectories for src/
add_subdirectory(src)
