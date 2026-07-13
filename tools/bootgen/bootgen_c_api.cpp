//===- bootgen_c_api.cpp - Exception-safe PDI generation API ----*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bootgen_c_api.h"
#include "bootgenexception.h"
#include "bootimage.h"
#include "options.h"

#include <cstring>
#include <string>

// Helper to safely copy error message to output buffer
static void copy_error_message(char *error_msg, int error_msg_size,
                               const char *msg) {
  if (error_msg && error_msg_size > 0) {
    std::strncpy(error_msg, msg, error_msg_size - 1);
    error_msg[error_msg_size - 1] = '\0';
  }
}

int bootgen_generate_pdi(const char *bif_path, const char *pdi_path,
                         enum bootgen_arch_type arch, int overwrite,
                         char *error_msg, int error_msg_size) {
  // Validate inputs
  if (!bif_path) {
    copy_error_message(error_msg, error_msg_size,
                       "Invalid NULL bif_path argument");
    return BOOTGEN_ERROR_INVALID_BIF;
  }
  if (!pdi_path) {
    copy_error_message(error_msg, error_msg_size,
                       "Invalid NULL pdi_path argument");
    return BOOTGEN_ERROR_INVALID_OUTPUT;
  }

  // Map C enum to C++ enum
  Arch::Type cppArch;
  switch (arch) {
  case BOOTGEN_ARCH_ZYNQ:
    cppArch = Arch::ZYNQ;
    break;
  case BOOTGEN_ARCH_ZYNQMP:
    cppArch = Arch::ZYNQMP;
    break;
  case BOOTGEN_ARCH_FPGA:
    cppArch = Arch::FPGA;
    break;
  case BOOTGEN_ARCH_VERSAL:
    cppArch = Arch::VERSAL;
    break;
  case BOOTGEN_ARCH_VERSALNET:
    cppArch = Arch::VERSALNET;
    break;
  default:
    copy_error_message(error_msg, error_msg_size, "Invalid architecture type");
    return BOOTGEN_ERROR_INVALID_ARCH;
  }

  try {
    Options options;
    options.SetArchType(cppArch);
    options.SetBifFilename(std::string(bif_path));
    options.InsertOutputFileNames(std::string(pdi_path));
    if (overwrite) {
      options.SetOverwrite(true);
    }

    std::string bifStr(bif_path);
    BIF_File bif(bifStr);
    bif.Process(options);

    return BOOTGEN_SUCCESS;
  } catch (const BootGenExceptionClass &e) {
    copy_error_message(error_msg, error_msg_size, e.what());
    return BOOTGEN_ERROR_PROCESSING_FAILED;
  } catch (const std::exception &e) {
    copy_error_message(error_msg, error_msg_size, e.what());
    return BOOTGEN_ERROR_INTERNAL;
  } catch (...) {
    copy_error_message(error_msg, error_msg_size, "Unknown error occurred");
    return BOOTGEN_ERROR_INTERNAL;
  }
}
