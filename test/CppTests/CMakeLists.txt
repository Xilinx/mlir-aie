# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_executable(target_model  target_model.cpp)
add_executable(target_model_rtti  target_model_rtti.cpp)
add_executable(target_model_unknown_device  target_model_unknown_device.cpp)
add_executable(register_database  register_database.cpp)
add_test(NAME TargetModel COMMAND target_model)
add_test(NAME TargetModelRtti COMMAND target_model_rtti)
add_test(NAME TargetModelUnknownDevice COMMAND target_model_unknown_device)
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

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" AIE_CPP_TEST_HOST_PROCESSOR)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   AIE_CPP_TEST_HOST_PROCESSOR MATCHES "^(x86_64|amd64|i[3-6]86|x86)$")
  llvm_map_components_to_libnames(stack_analysis_llvm_libs support core irreader object)

  add_executable(stack_size_analysis_false_recursion_test
    stack_size_analysis_false_recursion_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/aiecc/StackSizeAnalysis.cpp)
  target_include_directories(stack_size_analysis_false_recursion_test
                             PRIVATE ${CMAKE_SOURCE_DIR})
  target_compile_definitions(stack_size_analysis_false_recursion_test
                             PRIVATE
                             AIE_STACK_SIZE_ANALYSIS_TEST_CLANG="${CMAKE_C_COMPILER}")
  target_link_libraries(stack_size_analysis_false_recursion_test
                        PRIVATE ${stack_analysis_llvm_libs})
  add_test(NAME StackSizeAnalysisFalseRecursion
           COMMAND stack_size_analysis_false_recursion_test)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(EXECUTABLES target_model target_model_rtti target_model_unknown_device register_database)

set(CPP_TEST_DEPENDS ${EXECUTABLES} bfloat16_fallback_test)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   AIE_CPP_TEST_HOST_PROCESSOR MATCHES "^(x86_64|amd64|i[3-6]86|x86)$")
  list(APPEND CPP_TEST_DEPENDS stack_size_analysis_false_recursion_test)
endif()
add_custom_target(check-aie-cpp COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS ${CPP_TEST_DEPENDS})

foreach(executable ${EXECUTABLES})
    target_link_libraries(${executable}
                        PUBLIC
                        AIE
                        ${dialect_libs})
endforeach()

add_dependencies(check-aie check-aie-cpp)
