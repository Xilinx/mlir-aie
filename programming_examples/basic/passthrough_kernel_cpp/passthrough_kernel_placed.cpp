//===- passthrough_kernel_placed.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

namespace {

// Command line option parser
struct Options {
  std::string device = "npu";
  int64_t in1Size = 4096;
  int64_t outSize = 4096;
  int64_t traceSize = 0;

  bool parseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
        device = argv[++i];
      } else if ((arg == "-i1s" || arg == "--in1_size") && i + 1 < argc) {
        in1Size = std::stoll(argv[++i]);
      } else if ((arg == "-os" || arg == "--out_size") && i + 1 < argc) {
        outSize = std::stoll(argv[++i]);
      } else if ((arg == "-t" || arg == "--trace_size") && i + 1 < argc) {
        traceSize = std::stoll(argv[++i]);
      } else if (arg == "-h" || arg == "--help") {
        std::cout << "Usage: " << argv[0] << " [options]\n";
        std::cout << "Options:\n";
        std::cout << "  -d, --device <npu|npu2>   AIE Device (default: npu)\n";
        std::cout << "  -i1s, --in1_size <bytes>  Input size in bytes (default: 4096)\n";
        std::cout << "  -os, --out_size <bytes>   Output size in bytes (default: 4096)\n";
        std::cout << "  -t, --trace_size <bytes>  Trace buffer size (default: 0)\n";
        std::cout << "  -h, --help                Show this help\n";
        return false;
      }
    }
    return true;
  }
};

// Main function to generate the passthrough kernel MLIR
void generatePassthroughKernel(const std::string &device, int64_t in1Size,
                                int64_t outSize, int64_t traceSize) {
  // Calculate sizes
  int64_t N = in1Size; // N elements of uint8
  int64_t lineWidthInBytes = N / 4; // chop input in 4 sub-tensors

  // Check that output size matches input size
  if (outSize != in1Size) {
    std::cerr << "Error: Output buffer size must be equal to input buffer "
                    "size.\n";
    exit(1);
  }

  // Check input size constraints
  if (in1Size % 64 != 0 || in1Size < 512) {
    std::cerr << "Error: In1 buffer size (" << in1Size
                 << ") must be a multiple of 64 and greater than or equal to "
                    "512\n";
    exit(1);
  }

  // Note: Tracing support (trace_size parameter) not yet implemented
  // The Python version uses trace_utils to configure packet tracing flows
  // This could be added in a future enhancement
  if (traceSize > 0) {
    std::cerr << "Warning: Trace support not yet implemented in C++ version\n";
  }

  // Determine device enum
  std::string deviceEnum;
  if (device == "npu") {
    deviceEnum = "npu1_1col";
  } else if (device == "npu2") {
    deviceEnum = "npu2";
  } else {
    std::cerr << "Error: Device name " << device << " is unknown\n";
    exit(1);
  }

  // Generate MLIR text
  std::cout << "module {\n";
  std::cout << "  aie.device(" << deviceEnum << ") {\n";
  std::cout << "    %tile_0_0 = aie.tile(0, 0)\n";
  std::cout << "    %tile_0_2 = aie.tile(0, 2)\n";
  std::cout << "    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<" << lineWidthInBytes << "xi8>>\n";
  std::cout << "    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<" << lineWidthInBytes << "xi8>>\n";
  std::cout << "    func.func private @passThroughLine(memref<" << lineWidthInBytes << "xi8>, memref<" << lineWidthInBytes << "xi8>, i32)\n";
  std::cout << "    %core_0_2 = aie.core(%tile_0_2) {\n";
  std::cout << "      %c0 = arith.constant 0 : index\n";
  std::cout << "      %c" << std::numeric_limits<int32_t>::max() << " = arith.constant " << std::numeric_limits<int32_t>::max() << " : index\n";
  std::cout << "      %c1 = arith.constant 1 : index\n";
  std::cout << "      scf.for %arg0 = %c0 to %c" << std::numeric_limits<int32_t>::max() << " step %c1 {\n";
  std::cout << "        %0 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<" << lineWidthInBytes << "xi8>>\n";
  std::cout << "        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<" << lineWidthInBytes << "xi8>> -> memref<" << lineWidthInBytes << "xi8>\n";
  std::cout << "        %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<" << lineWidthInBytes << "xi8>>\n";
  std::cout << "        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<" << lineWidthInBytes << "xi8>> -> memref<" << lineWidthInBytes << "xi8>\n";
  std::cout << "        %c" << lineWidthInBytes << "_i32 = arith.constant " << lineWidthInBytes << " : i32\n";
  std::cout << "        func.call @passThroughLine(%3, %1, %c" << lineWidthInBytes << "_i32) : (memref<" << lineWidthInBytes << "xi8>, memref<" << lineWidthInBytes << "xi8>, i32) -> ()\n";
  std::cout << "        aie.objectfifo.release @in(Consume, 1)\n";
  std::cout << "        aie.objectfifo.release @out(Produce, 1)\n";
  std::cout << "      }\n";
  std::cout << "      aie.end\n";
  std::cout << "    } {link_with = \"passThrough.cc.o\"}\n";
  std::cout << "    func.func @sequence(%arg0: memref<" << N << "xi8>, %arg1: memref<" << N << "xi8>, %arg2: memref<" << N << "xi8>) {\n";
  std::cout << "      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, " << N << "][0, 0, 0]) {id = 0 : i64, metadata = @in} : memref<" << N << "xi8>\n";
  std::cout << "      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, " << N << "][0, 0, 0]) {id = 1 : i64, metadata = @out} : memref<" << N << "xi8>\n";
  std::cout << "      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}\n";
  std::cout << "      return\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
  std::cout << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!opts.parseArgs(argc, argv)) {
    return 0;
  }

  // Generate the kernel
  generatePassthroughKernel(opts.device, opts.in1Size, opts.outSize, opts.traceSize);

  return 0;
}
