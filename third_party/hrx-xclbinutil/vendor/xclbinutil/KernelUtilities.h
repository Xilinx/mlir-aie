/**
 * Copyright (C) 2021-2022 Xilinx, Inc
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef __KernelUtilities_h_
#define __KernelUtilities_h_

// Please keep the include files to a minimum
#include "hrx_util.h"
#include <string>
#include <vector>

namespace XclBinUtilities {
void addKernel(const hrx::property_tree::ptree& ptKernel, bool isFixedPS, hrx::property_tree::ptree& ptEmbeddedData);
void addKernel(const hrx::property_tree::ptree& ptKernel, hrx::property_tree::ptree& ptMemTopology, hrx::property_tree::ptree& ptIPLayout, hrx::property_tree::ptree& ptConnectivity);
void validateFunctions(const std::string& kernelLibrary, const hrx::property_tree::ptree& ptFunctions);
void createPSKernelMetadata(const std::string& memBank, unsigned long numInstances, const hrx::property_tree::ptree& ptFunctions, const std::string& kernelLibrary, hrx::property_tree::ptree& ptPSKernels);
};
#endif
