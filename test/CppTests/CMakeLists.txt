# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model  target_model.cpp)
add_executable(target_model_rtti  target_model_rtti.cpp)
add_executable(register_database  register_database.cpp)
add_test(NAME TargetModel COMMAND target_model)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)
add_test(NAME RegisterDatabase COMMAND register_database WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Exercises the uint16 fallback path of test_utils::bfloat16_t on every
# platform by force-undefining __STDCPP_BFLOAT16_T__ for this TU only.
add_executable(bfloat16_fallback_test bfloat16_fallback_test.cpp)
target_include_directories(bfloat16_fallback_test
                           PRIVATE ${CMAKE_SOURCE_DIR}/runtime_lib/test_lib)
if(MSVC)
  target_compile_options(bfloat16_fallback_test PRIVATE /U__STDCPP_BFLOAT16_T__)
else()
  target_compile_options(bfloat16_fallback_test PRIVATE -U__STDCPP_BFLOAT16_T__)
endif()
add_test(NAME Bfloat16Fallback COMMAND bfloat16_fallback_test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(EXECUTABLES target_model target_model_rtti register_database)

add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${EXECUTABLES} bfloat16_fallback_test)

foreach(executable ${EXECUTABLES})
    target_link_libraries(${executable}
                        PUBLIC
                        AIE
                        ${dialect_libs})
endforeach()

add_dependencies(check-aie check-aie-cpp)
