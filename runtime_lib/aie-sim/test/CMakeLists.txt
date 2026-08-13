#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Plain executables plus CTest, matching the convention in test/CppTests.
# Every test here runs on any machine: no Vitis, no license, no NPU.

set(AIE_SIM_TESTS
  device_test
  memory_test
  readings_test
  region_map_test
  elf_loader_test
  cdo_replay_test
  lock_test
  dma_test
  stream_switch_test
  shim_mux_test
  scheduler_test
  core_window_test
  # The only test that crosses into llvm-aie. Reports SKIP rather than PASS
  # when no engine is configured, since the engine is built by that project.
  core_engine_test
  core_register_map_test
  tile_core_port_test
  core_dispatch_test
  core_exec_test
  core_datapath_test
  core_compiled_test
  core_lock_test
  core_stall_test
  # Guards the component's central claim: that the ess_* symbols aie-rt
  # declares are all provided, with matching signatures. Compiling it is the
  # signature check, linking it is the completeness check.
  ess_abi_test
  # Drives the model through the real vendored aie-rt driver instead of
  # hand-packed registers (see the aie_rt_sim target in the parent
  # CMakeLists.txt). Optional: skipped below if that target is not available.
  aie_rt_integration_test
)

set(AIE_SIM_PRESENT_TESTS)
foreach(t ${AIE_SIM_TESTS})
  if(t STREQUAL "aie_rt_integration_test" AND NOT AIE_SIM_HAVE_AIE_RT)
    message(STATUS "aie-sim: aie-rt not available, skipping ${t}")
    continue()
  endif()
  # While the simulator is being built up, a component whose test has not been
  # written yet simply contributes no target. Once all of them exist this loop
  # is unconditional; until then a missing file must not break the build for
  # everyone else.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
    add_executable(${t} ${t}.cpp)
    if(t STREQUAL "aie_rt_integration_test")
      # aie_rt_sim first: its undefined ess_* symbols (Array.h's ABI) are
      # resolved from aie-sim, and a plain ld does one left-to-right pass
      # over static libraries, so the library WITH the undefined refs must
      # come before the one that satisfies them.
      target_link_libraries(${t} PRIVATE aie_rt_sim aie-sim)
    else()
      target_link_libraries(${t} PRIVATE aie-sim)
    endif()
    add_test(NAME aie-sim-${t} COMMAND ${t})
    if(t STREQUAL "core_engine_test" OR t STREQUAL "core_dispatch_test" OR t STREQUAL "core_exec_test" OR t STREQUAL "core_datapath_test" OR t STREQUAL "core_compiled_test" OR t STREQUAL "core_lock_test" OR t STREQUAL "core_stall_test")
      # Exit 77 means "no engine configured", which must read as SKIP. Letting
      # it pass would report a green boundary test with nothing on the far side.
      set_tests_properties(aie-sim-${t} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    list(APPEND AIE_SIM_PRESENT_TESTS ${t})
  else()
    message(STATUS "aie-sim: ${t}.cpp not present yet, skipping")
  endif()
endforeach()

add_custom_target(check-aie-sim
                  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                  DEPENDS ${AIE_SIM_PRESENT_TESTS})
