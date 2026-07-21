# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# FindHRX.cmake - locate the HRX (libhrx amdxdna) runtime for the IRON host stack.
#
# Helps locate the HRX runtime so callers don't have to hard-code paths or rely on
# activate_env.sh. Environment variables, when set, are treated only as *hints*
# with the highest priority -- detection still works without them.
#
# Result variables:
#   HRX_FOUND                    TRUE if a usable HRX runtime was located
#                                (headers + libhrx.so).
#   HRX_DIR                      Root of the HRX install prefix (or source tree).
#   HRX_INCLUDE_DIR              Dir containing hrx_runtime.h + hrx_amdxdna.h.
#   HRX_LIBHRX                   Full path to libhrx.so.
#   HRX_LIBHRX_DIR               Dir containing libhrx.so.
#
# Components (find_package(HRX COMPONENTS runtime)):
#   runtime                      headers + libhrx.so (enough to link; the default
#                                when no COMPONENTS are requested).
#   Sets HRX_runtime_FOUND accordingly.
#
# Imported target:
#   HRX::hrx                     INTERFACE target carrying HRX_INCLUDE_DIR +
#                                libhrx.so. Prefer linking this over the raw
#                                HRX_LIBHRX / HRX_INCLUDE_DIR variables. When the
#                                shipped CMake package is used, HRX::hrx is an
#                                alias of the package's hrx::hrx target.

include(FindPackageHandleStandardArgs)

# --- candidate roots (env hints first, then standard locations) --------------
# Only generic, upstreamable locations here -- no developer-specific paths. Set
# HRX_DIR to point at a non-standard install prefix.
set(_hrx_root_hints
  "$ENV{HRX_DIR}"
  # A sibling ../hrx-system install/build next to mlir-aie (kept in sync with
  # the Python discovery in python/utils/hostruntime/hrxruntime/discovery.py).
  "${CMAKE_CURRENT_LIST_DIR}/../../../hrx-system/build/hrx-install"
  "${CMAKE_CURRENT_LIST_DIR}/../../../../hrx-system/build/hrx-install"
  "${CMAKE_CURRENT_LIST_DIR}/../../../hrx"
  "${CMAKE_CURRENT_LIST_DIR}/../../../../hrx"
  "$ENV{HOME}/hrx"
  "/opt/hrx"
  "/usr/local/hrx"
)

# --- 0) prefer the shipped CMake package (hrx-config.cmake -> hrx::hrx) -------
# find_package(hrx CONFIG) resolves an install prefix on CMAKE_PREFIX_PATH or an
# HRX_DIR hint. When it succeeds we adopt its hrx::hrx target and derive the
# include/lib result variables from it.
if(NOT TARGET hrx::hrx)
  set(_hrx_config_hints "")
  foreach(_h IN LISTS _hrx_root_hints)
    if(_h)
      list(APPEND _hrx_config_hints "${_h}")
    endif()
  endforeach()
  find_package(hrx CONFIG QUIET HINTS ${_hrx_config_hints})
endif()

if(TARGET hrx::hrx)
  # Derive the public include dir + libhrx.so from the packaged target so the
  # legacy HRX_* variables (and the HRX::hrx alias below) keep working.
  get_target_property(_hrx_pkg_incs hrx::hrx INTERFACE_INCLUDE_DIRECTORIES)
  if(_hrx_pkg_incs)
    foreach(_inc IN LISTS _hrx_pkg_incs)
      if(EXISTS "${_inc}/hrx_runtime.h")
        set(HRX_INCLUDE_DIR "${_inc}" CACHE PATH "Directory containing hrx_runtime.h" FORCE)
      endif()
    endforeach()
  endif()
  get_target_property(_hrx_pkg_loc hrx::hrx LOCATION)
  if(_hrx_pkg_loc AND EXISTS "${_hrx_pkg_loc}")
    set(HRX_LIBHRX "${_hrx_pkg_loc}" CACHE FILEPATH "Path to libhrx.so" FORCE)
  endif()
endif()

# --- 1) the public header (anchors the HRX root) -----------------------------
# Handles both a normal install prefix (<root>/include/hrx) and the in-tree
# source-checkout layout (<root>/libhrx/include).
if(NOT HRX_INCLUDE_DIR)
  find_path(HRX_INCLUDE_DIR
    NAMES hrx_runtime.h
    HINTS ${_hrx_root_hints}
    PATH_SUFFIXES include/hrx include libhrx/include
    DOC "Directory containing hrx_runtime.h"
  )
endif()

if(HRX_INCLUDE_DIR)
  if(EXISTS "${HRX_INCLUDE_DIR}/../../libhrx/include/hrx_runtime.h")
    # <root>/libhrx/include/hrx_runtime.h -> <root>  (source checkout)
    get_filename_component(HRX_DIR "${HRX_INCLUDE_DIR}/../.." ABSOLUTE)
  elseif(HRX_INCLUDE_DIR MATCHES "/include/hrx$")
    # <root>/include/hrx/hrx_runtime.h -> <root>  (install prefix, headers in hrx/)
    get_filename_component(HRX_DIR "${HRX_INCLUDE_DIR}/../.." ABSOLUTE)
  else()
    # <root>/include/hrx_runtime.h -> <root>  (flat install prefix)
    get_filename_component(HRX_DIR "${HRX_INCLUDE_DIR}/.." ABSOLUTE)
  endif()
endif()

# --- 2) libhrx.so ------------------------------------------------------------
if(NOT HRX_LIBHRX)
  find_library(HRX_LIBHRX
    NAMES hrx libhrx
    HINTS
      "$ENV{LIBHRX_DIR}"
      "${HRX_DIR}/lib"                                   # install-prefix layout
      "${HRX_DIR}/build/cmake/libhrx/src/libhrx"         # source-build layout
      "${HRX_DIR}/build/hrx-install/lib"
    PATHS /usr/lib /usr/local/lib
    DOC "Path to libhrx.so"
  )
endif()

if(HRX_LIBHRX)
  get_filename_component(HRX_LIBHRX_DIR "${HRX_LIBHRX}" DIRECTORY)
endif()

# --- components --------------------------------------------------------------
if(HRX_INCLUDE_DIR AND HRX_LIBHRX)
  set(HRX_runtime_FOUND TRUE)
else()
  set(HRX_runtime_FOUND FALSE)
endif()
if(NOT HRX_FIND_COMPONENTS)
  set(HRX_FIND_COMPONENTS runtime)
  set(HRX_FIND_REQUIRED_runtime ${HRX_FIND_REQUIRED})
endif()

find_package_handle_standard_args(HRX
  REQUIRED_VARS HRX_INCLUDE_DIR HRX_LIBHRX
  HANDLE_COMPONENTS
  FAIL_MESSAGE
    "Could not find HRX. Set HRX_DIR to an HRX install prefix (with include/hrx/hrx_runtime.h + lib/libhrx.so and lib/cmake/hrx), add that prefix to CMAKE_PREFIX_PATH, or set LIBHRX_DIR (dir with libhrx.so)."
)

# --- imported target ---------------------------------------------------------
# Consumers should link HRX::hrx rather than the raw HRX_LIBHRX / HRX_INCLUDE_DIR
# variables. When the shipped CMake package provided hrx::hrx we simply alias it;
# otherwise we synthesize an INTERFACE target from the discovered include/lib.
if(HRX_FOUND AND NOT TARGET HRX::hrx)
  add_library(HRX::hrx INTERFACE IMPORTED)
  if(TARGET hrx::hrx)
    set_target_properties(HRX::hrx PROPERTIES INTERFACE_LINK_LIBRARIES "hrx::hrx")
  else()
    set_target_properties(HRX::hrx PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HRX_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${HRX_LIBHRX}"
    )
  endif()
endif()

mark_as_advanced(HRX_INCLUDE_DIR HRX_LIBHRX)
