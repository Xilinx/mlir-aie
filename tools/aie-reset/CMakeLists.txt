#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_executable(aie-reset aie-reset.cpp)

include_directories(${XILINX_XAIE_INCLUDE_DIR})
link_libraries(${XILINX_XAIE_LIBS})

install(TARGETS aie-reset
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-reset)
