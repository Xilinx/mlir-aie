#
# Copyright (C) 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(aie-reset aie-reset.cpp)

install(TARGETS aie-reset
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-reset)
