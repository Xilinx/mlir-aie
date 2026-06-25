//===- txn_encoding_host.cpp - Standalone TxnEncoding.h byte test ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Exercises include/aie/Runtime/TxnEncoding.h the way a downstream, dependency-
// free consumer (e.g. generated host C++) does: include only this header, build
// a transaction from plain integers, and print the resulting words. The output
// is byte-checked by the accompanying .test file so the wire format is pinned
// independently of the MLIR translate path.
//
//===----------------------------------------------------------------------===//

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace aie_runtime;

int main() {
  std::vector<uint32_t> txn;
  txn_init(txn);

  uint32_t count = 0;

  txn_append_write32(txn, /*addr=*/0x0006400DEFu, /*val=*/0x42u);
  ++count;

  txn_append_maskwrite32(txn, /*addr=*/0x00012340u, /*val=*/0xABu,
                         /*mask=*/0xFFu);
  ++count;

  txn_append_sync(txn, /*col=*/1, /*row=*/2, /*dir=*/0, /*chan=*/3,
                  /*ncol=*/4, /*nrow=*/5);
  ++count;

  const uint32_t blk[] = {0xDEADBEEFu, 0x12345678u, 0x9ABCDEF0u};
  txn_append_blockwrite(txn, /*addr=*/0x00021000u, blk, /*count=*/3,
                        /*col=*/6, /*row=*/7);
  ++count;

  txn_append_address_patch(txn, /*addr=*/0x00033000u, /*arg_idx=*/2,
                           /*arg_plus=*/0x80u);
  ++count;

  txn_append_loadpdi(txn, /*id=*/9, /*size=*/0x100u,
                     /*addr=*/0x00000001DEADC0DEull);
  ++count;

  txn_append_preempt(txn, /*level=*/3);
  ++count;

  TxnDeviceInfo info; // defaults: NPU (devGen=3)
  info.devGen = 4;    // NPU2
  info.numCols = 5;
  info.numRows = 6;
  info.numMemTileRows = 1;
  txn_prepend_header(txn, count, info);

  for (uint32_t w : txn)
    printf("%08X\n", w);
  return 0;
}
