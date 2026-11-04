# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generates AIECCVersion.h with the current git SHA and tracked-source dirty status.
# Only rewrites the file when the identity changes, so aiecc.cpp is not pointlessly rebuilt.
#
# Expects:
#   SOURCE_DIR  - repository source dir (to query git)
#   OUTPUT_FILE - path of the AIECCVersion.h header to (re)generate

if(NOT DEFINED SOURCE_DIR)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(AIECC_GIT_SHA "unknown")
set(AIECC_GIT_DIRTY -1)
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE _git_sha
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_git_sha)
    set(AIECC_GIT_SHA "${_git_sha}")
    # The build applies repository-owned patches inside dependency submodules.
    # Their revisions still count; their patched working trees do not.
    execute_process(
      COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD --ignore-submodules=dirty --
      WORKING_DIRECTORY ${SOURCE_DIR}
      RESULT_VARIABLE _dirty
      ERROR_QUIET)
    if(_dirty EQUAL 0 OR _dirty EQUAL 1)
      set(AIECC_GIT_DIRTY ${_dirty})
    endif()
  endif()
endif()

set(_new_content "#pragma once\n#define AIECC_GIT_SHA \"${AIECC_GIT_SHA}\"\n#define AIECC_GIT_DIRTY ${AIECC_GIT_DIRTY}\n")

set(_old_content "")
if(EXISTS ${OUTPUT_FILE})
  file(READ ${OUTPUT_FILE} _old_content)
endif()

if(NOT _old_content STREQUAL _new_content)
  file(WRITE ${OUTPUT_FILE} "${_new_content}")
endif()
