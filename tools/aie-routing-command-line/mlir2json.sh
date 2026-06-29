# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

aie-opt --aie-create-pathfinder-flows --aie-find-flows $1 | aie-translate --aie-flows-to-json > $2.json
