# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Applies each PATCH_DIR/*.patch to the git tree at ROOT. A patch that a reverse
# dry run says is already applied is skipped, so reconfiguring is a no-op.
function(apply_vendored_patches ROOT PATCH_DIR)
  file(GLOB _patches ${PATCH_DIR}/*.patch)
  list(SORT _patches)
  find_package(Git REQUIRED)
  foreach(_patch ${_patches})
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${_patch}
      WORKING_DIRECTORY ${ROOT}
      RESULT_VARIABLE _already_applied
      ERROR_QUIET)
    if(_already_applied EQUAL 0)
      message(STATUS "Vendored patch already applied, skipping: ${_patch}")
      continue()
    endif()

    message(STATUS "Applying vendored patch: ${_patch}")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply ${_patch}
      WORKING_DIRECTORY ${ROOT}
      RESULT_VARIABLE _patch_result
      ERROR_VARIABLE _patch_error)
    if(NOT _patch_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to apply vendored patch ${_patch}:\n${_patch_error}\n"
        "If ${ROOT} has local modifications, reset it with "
        "'git -C ${ROOT} checkout -- .' and re-run CMake.")
    endif()
  endforeach()
endfunction()
