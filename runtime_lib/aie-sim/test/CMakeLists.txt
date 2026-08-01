#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Plain executables plus CTest, matching the convention in test/CppTests.
# Every test here runs on any machine: no Vitis, no license, no NPU.

set(AIE_SIM_TESTS
  device_test
  memory_test
  lock_test
  dma_test
  stream_switch_test
  scheduler_test
  # Guards the component's central claim: that the ess_* symbols aie-rt
  # declares are all provided, with matching signatures. Compiling it is the
  # signature check, linking it is the completeness check.
  ess_abi_test
)

set(AIE_SIM_PRESENT_TESTS)
foreach(t ${AIE_SIM_TESTS})
  # While the simulator is being built up, a component whose test has not been
  # written yet simply contributes no target. Once all of them exist this loop
  # is unconditional; until then a missing file must not break the build for
  # everyone else.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE aie-sim)
    add_test(NAME aie-sim-${t} COMMAND ${t})
    list(APPEND AIE_SIM_PRESENT_TESTS ${t})
  else()
    message(STATUS "aie-sim: ${t}.cpp not present yet, skipping")
  endif()
endforeach()

add_custom_target(check-aie-sim
                  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                  DEPENDS ${AIE_SIM_PRESENT_TESTS})
