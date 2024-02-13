////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_EXT_AIR_H_
#define HSA_RUNTIME_EXT_AIR_H_

#include <stdint.h>

#define AIR_ADDRESS_ABSOLUTE 0x0L
#define AIR_ADDRESS_ABSOLUTE_RANGE 0x1L
#define AIR_ADDRESS_HERD_RELATIVE 0x2L
#define AIR_ADDRESS_HERD_RELATIVE_RANGE 0x3L

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief AIR vendor-specific packet type.
 */
typedef enum {
  AIR_PKT_TYPE_INVALID = 0,
  AIR_PKT_TYPE_PUT_STREAM = 1,
  AIR_PKT_TYPE_GET_STREAM = 2,
  AIR_PKT_TYPE_SDMA_STATUS = 3,
  AIR_PKT_TYPE_TDMA_STATUS = 4,
  AIR_PKT_TYPE_CORE_STATUS = 5,

  AIR_PKT_TYPE_DEVICE_INITIALIZE = 0x0010L,
  AIR_PKT_TYPE_SEGMENT_INITIALIZE = 0x0011L,
  AIR_PKT_TYPE_HELLO = 0x0012L,
  AIR_PKT_TYPE_ALLOCATE_HERD_SHIM_DMAS = 0x0013L,
  AIR_PKT_TYPE_GET_CAPABILITIES = 0x0014L,
  AIR_PKT_TYPE_GET_INFO = 0x0015L,

  AIR_PKT_TYPE_XAIE_LOCK = 0x0020L,

  AIR_PKT_TYPE_CDMA = 0x030L,
  AIR_PKT_TYPE_CONFIGURE = 0x031L,

  AIR_PKT_TYPE_POST_RDMA_WQE = 0x040L,
  AIR_PKT_TYPE_POST_RDMA_RECV = 0x041L,

  AIR_PKT_TYPE_PROG_FIRMWARE = 0x050L,
  AIR_PKT_TYPE_READ_AIE_REG32 = 0x51L,
  AIR_PKT_TYPE_WRITE_AIE_REG32 = 0x52L,
  AIR_PKT_TYPE_AIRBIN = 0x53L,
  AIR_PKT_TYPE_TRANSLATE = 0x54L,

  AIR_PKT_TYPE_SHIM_DMA_MEMCPY = 0x0100L,
  AIR_PKT_TYPE_HERD_SHIM_DMA_MEMCPY = 0x0101L,
  AIR_PKT_TYPE_HERD_SHIM_DMA_1D_STRIDED_MEMCPY = 0x0102L,
  AIR_PKT_TYPE_ND_MEMCPY = 0x0103L,

} hsa_air_packet_type_t;

/**
 * @brief AIR agent attributes.
 */
typedef enum {
  AIR_AGENT_INFO_NAME = 0,
  AIR_AGENT_INFO_VENDOR_NAME = 1,
  AIR_AGENT_INFO_CONTROLLER_ID = 2,
  AIR_AGENT_INFO_FIRMWARE_VER = 3,
  AIR_AGENT_INFO_NUM_REGIONS = 4,
  AIR_AGENT_INFO_HERD_SIZE = 5,
  AIR_AGENT_INFO_HERD_ROWS = 6,
  AIR_AGENT_INFO_HERD_COLS = 7,
  AIR_AGENT_INFO_TILE_DATA_MEM_SIZE = 8,
  AIR_AGENT_INFO_TILE_PROG_MEM_SIZE = 9,
  AIR_AGENT_INFO_L2_MEM_SIZE = 10
} hsa_air_agent_info_t;

#ifdef __cplusplus
} // end extern "C" block
#endif

#endif // header guard
