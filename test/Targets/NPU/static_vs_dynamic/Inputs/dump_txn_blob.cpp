//===- dump_txn_blob.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call a generated TXN builder and write the returned word stream to a file as
// raw little-endian bytes -- exactly the blob a host app would hand to the
// downstream ELF packager. The driver test then runs `aiebu-asm -t aie2txn` on
// this file to assert the stream satisfies aiebu's block-write-covers-patch
// invariant (which plain XRT dispatch does not enforce).
//
// Two modes, selected at compile time:
//   * default: the real builder GEN_FN (#included via -DGEN_HDR), invoked with
//     ARGVAL, is written out. GEN_FN / ARGVAL come from -D so this one file
//     serves any dynamic sequence. This is the POSITIVE case aiebu must accept.
//   * -DMAKE_BAD: emit a hand-built stream with a DDR_PATCH that no preceding
//     block-write covers -- the shape the pre-fix dynamic lowering produced.
//     This is the NEGATIVE case aiebu must reject, proving the test detects a
//     violation rather than passing vacuously. It needs no generated header.
//
// argv[1] is the output path.
//
//===----------------------------------------------------------------------===//

#ifndef MAKE_BAD
#include GEN_HDR
#endif

#include "aie/Runtime/TxnEncoding.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

// Hand-build the pre-fix-shaped violation: a DDR_PATCH into a BD register block
// with NO block-write covering that block. 0x1d000 is the shim BD base the real
// dynamic path targets, so aiebu's rejection message names the same address a
// genuine regression would.
std::vector<uint32_t> buildBadStream() {
  using namespace aie_runtime;
  std::vector<uint32_t> txn;
  txn_init(txn);
  txn_append_address_patch(txn, /*addr=*/0x1d000, /*arg_idx=*/0,
                           /*arg_plus=*/0);
  txn_prepend_header(txn, /*op_count=*/1u, {0, 1, 4, 6, 8, 1});
  return txn;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <out.bin>\n", argv[0]);
    return 2;
  }

#ifdef MAKE_BAD
  std::vector<uint32_t> txn = buildBadStream();
#else
  auto txnOpt = GEN_FN(ARGVAL);
  if (!txnOpt) {
    // A nullopt means the builder declined (e.g. a runtime scalar overflowed a
    // narrow BD field); for the invariant test the args are in range, so this
    // is a harness failure, not a valid empty stream.
    std::fprintf(stderr, "builder returned nullopt for arg=%d\n",
                 (int)(ARGVAL));
    return 2;
  }
  const std::vector<uint32_t> &txn = *txnOpt;
#endif

  std::ofstream out(argv[1], std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "cannot open output '%s'\n", argv[1]);
    return 2;
  }
  out.write(reinterpret_cast<const char *>(txn.data()),
            static_cast<std::streamsize>(txn.size() * sizeof(uint32_t)));
  std::fprintf(stderr, "wrote %zu words (%zu bytes) to %s\n", txn.size(),
               txn.size() * sizeof(uint32_t), argv[1]);
  return 0;
}
