# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model  target_model.cpp)
add_executable(target_model_rtti  target_model_rtti.cpp)
add_executable(target_model_unknown_device  target_model_unknown_device.cpp)
add_executable(register_database  register_database.cpp)
add_executable(register_field_test  register_field_test.cpp)
add_test(NAME TargetModel COMMAND target_model)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)
add_test(NAME TargetModelUnknownDevice COMMAND target_model_unknown_device)
add_test(NAME RegisterDatabase COMMAND register_database WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME RegisterField COMMAND register_field_test)

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

set(EXECUTABLES target_model target_model_rtti target_model_unknown_device register_database register_field_test)

add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${EXECUTABLES} bfloat16_fallback_test)

foreach(executable ${EXECUTABLES})
    target_link_libraries(${executable}
                        PUBLIC
                        AIE
                        ${dialect_libs})
endforeach()

# register_field_test exercises AIEX::encodeRegisterField directly (see
# lib/Dialect/AIEX/Utils/RegisterField.cpp), which lives in AIEXUtils rather
# than the core AIE dialect library the other executables above test.
target_link_libraries(register_field_test PUBLIC AIEXUtils)

add_dependencies(check-aie check-aie-cpp)
