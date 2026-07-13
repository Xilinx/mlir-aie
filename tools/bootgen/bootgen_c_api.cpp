/******************************************************************************
 * Copyright 2024-2026 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "bootgen_c_api.h"
#include "bootimage.h"
#include "options.h"
#include "bootgenexception.h"

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
  if (!bif_path || !pdi_path) {
    copy_error_message(error_msg, error_msg_size,
                       "Invalid NULL path argument");
    return BOOTGEN_ERROR_INVALID_BIF;
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
