# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model  target_model.cpp)
add_executable(target_model_rtti  target_model_rtti.cpp)
add_test(NAME TargetModel COMMAND target_model)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(EXECUTABLES target_model target_model_rtti)

add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${EXECUTABLES})

foreach(executable ${EXECUTABLES})
    target_link_libraries(${executable}
                        PUBLIC
                        AIE
                        ${dialect_libs})
endforeach()

add_dependencies(check-aie check-aie-cpp)
