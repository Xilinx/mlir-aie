# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model_rtti  target_model_rtti.cpp)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)

add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS target_model_rtti)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(target_model_rtti
                      PUBLIC
                      AIE
                      ${dialect_libs})

add_dependencies(check-aie check-aie-cpp)
