/**
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2019-2021 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef _XRT_VERSION_H_
#define _XRT_VERSION_H_

static const char xrt_build_version[] = "2.20.0";

static const char xrt_build_version_branch[] = "master";

static const char xrt_build_version_hash[] =
    "6771f4d86fa616b8908d86d0999ee802074a32d8";

static const char xrt_build_version_hash_date[] =
    "Wed, 30 Apr 2025 16:34:04 -0700";

static const char xrt_build_version_date_rfc[] =
    "Thu, 01 May 2025 15:25:03 -0700";

static const char xrt_build_version_date[] = "2025-05-01 15:25:03";

static const char xrt_modified_files[] = "";

#define XRT_DRIVER_VERSION "2.20.0,6771f4d86fa616b8908d86d0999ee802074a32d8"

#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 20)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))
#define XRT_PATCH 0
#define XRT_HEAD_COMMITS 8219
#define XRT_BRANCH_COMMITS -1

#ifdef __cplusplus
#include <iostream>
#include <string>

namespace xrt::version {

inline void print(std::ostream &output) {
  output << "       XRT Build Version: " << xrt_build_version << std::endl;
  output << "    Build Version Branch: " << xrt_build_version_branch
         << std::endl;
  output << "      Build Version Hash: " << xrt_build_version_hash << std::endl;
  output << " Build Version Hash Date: " << xrt_build_version_hash_date
         << std::endl;
  output << "      Build Version Date: " << xrt_build_version_date_rfc
         << std::endl;

  std::string modified_files(xrt_modified_files);
  if (modified_files.empty())
    return;

  const std::string &delimiters = ","; // Our delimiter
  std::string::size_type last_pos = 0;
  int running_index = 1;
  while (last_pos < modified_files.length() + 1) {
    if (running_index == 1)
      output << "  Current Modified Files: ";
    else
      output << "                          ";

    output << running_index++ << ") ";

    auto pos = modified_files.find_first_of(delimiters, last_pos);

    if (pos == std::string::npos)
      pos = modified_files.length();

    output << modified_files.substr(last_pos, pos - last_pos) << std::endl;

    last_pos = pos + 1;
  }
}

} // namespace xrt::version
#endif // __cplusplus

#endif
