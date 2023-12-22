#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023, Advanced Micro Devices, Inc.

#set(DATA_FILES
#  generated-source/gen_cdo.h
#  generated-source/gen_cdo.cpp
#  generated-source/cdo_main.cpp
#)
#
#set(DATA_TARGETS ${DATA_FILES})
#list(TRANSFORM DATA_TARGETS PREPEND ${PROJECT_BINARY_DIR}/data/)
#
## copy data files to build area
#add_custom_target(data_file_build ALL DEPENDS ${DATA_TARGETS})
#
#foreach(file ${DATA_FILES})
#add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/data/${file}
#                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
#                   ${PROJECT_BINARY_DIR}/data/${file}
#                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
#endforeach()
#
#foreach(file ${DATA_FILES})
#  get_filename_component(DATA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/data/${file} PATH)
#  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${file} DESTINATION ${DATA_INSTALL_PREFIX})
#endforeach()
#


add_executable(cdo_main
  generated-source/gen_cdo.cpp generated-source/cdo_main.cpp
)
target_include_directories(cdo_main PUBLIC
  "/media/mlevental/9b03e9bc-780d-4e14-9b46-ec8089db51a6/Xilinx/Vitis/2023.2/aietools/include/drivers/aiengine"
  "/media/mlevental/9b03e9bc-780d-4e14-9b46-ec8089db51a6/Xilinx/Vitis/2023.2/aietools/include"
)
target_link_directories(cdo_main PUBLIC
  "/media/mlevental/9b03e9bc-780d-4e14-9b46-ec8089db51a6/Xilinx/Vitis/2023.2/aietools/lib/lnx64.o"
  "/home/mlevental/dev_projects/mlir-aie/cmake-build-debug/runtime_lib/x86_64/xaiengine/cdo"
)
target_link_libraries(cdo_main PUBLIC xaienginecdo cdo_driver)