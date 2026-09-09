# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# aie-translate traces each flow through the switchbox that configures it, so
# --aie-find-flows has to keep the interconnect configuration it recovers from.
# It also expects every flow to start on a core or DMA.
aie-opt --aie-create-pathfinder-flows \
        --aie-find-flows='keep-partial-flows=false remove-lifted=false' $1 \
  | aie-translate --aie-flows-to-json > $2.json
