//===- bootgen_c_api.h - Exception-safe PDI generation API ------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOOTGEN_C_API_H
#define BOOTGEN_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error codes returned by bootgen C API functions
 */
enum bootgen_error_code {
  BOOTGEN_SUCCESS = 0,
  BOOTGEN_ERROR_INVALID_ARCH = 1,
  BOOTGEN_ERROR_INVALID_BIF = 2,
  BOOTGEN_ERROR_INVALID_OUTPUT = 3,
  BOOTGEN_ERROR_PROCESSING_FAILED = 4,
  BOOTGEN_ERROR_INTERNAL = 5
};

/**
 * @brief Architecture types supported by bootgen
 */
enum bootgen_arch_type {
  BOOTGEN_ARCH_ZYNQ = 0,
  BOOTGEN_ARCH_ZYNQMP = 1,
  BOOTGEN_ARCH_FPGA = 2,
  BOOTGEN_ARCH_VERSAL = 3,
  BOOTGEN_ARCH_VERSALNET = 4
};

/**
 * @brief Generate a PDI file from a BIF file
 *
 * This function provides a C-compatible interface to bootgen's PDI generation
 * functionality. It catches any C++ exceptions internally and returns error
 * codes instead.
 *
 * @param bif_path Path to the BIF (Boot Image Format) file
 * @param pdi_path Output path for the generated PDI file
 * @param arch Architecture type (use BOOTGEN_ARCH_VERSAL for AIE)
 * @param overwrite If non-zero, overwrite existing output file
 * @param error_msg Optional buffer to receive error message (can be NULL)
 * @param error_msg_size Size of error_msg buffer
 *
 * @return BOOTGEN_SUCCESS on success, or an error code on failure
 */
int bootgen_generate_pdi(const char *bif_path, const char *pdi_path,
                         enum bootgen_arch_type arch, int overwrite,
                         char *error_msg, int error_msg_size);

#ifdef __cplusplus
}
#endif

#endif /* BOOTGEN_C_API_H */
