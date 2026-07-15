//===- compare_main.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Three-way TXN equivalence harness for the static-vs-dynamic tests.
//
// It proves the milestone #3222 correctness spine for one runtime sequence:
//
//   1. golden   = the production binary emitter (aie-translate --aie-npu-to-
//                 binary), read from the hex file passed as argv[1].
//   2. STATIC_FN = the C++ TXN builder generated from the *static* sequence
//                 (all-constant fields).
//   3. DYN_FN(ARGVAL) = the C++ TXN builder generated from the *dynamic*
//                 sequence, invoked with the same constant the static side
//                 bakes in.
//
// All three word streams must be byte-identical. Splitting the check three
// ways localizes a regression: golden-vs-STATIC isolates the EmitC codegen
// against the binary emitter; STATIC-vs-DYN isolates runtime-argument
// substitution.
//
// The generated header (which defines STATIC_FN and DYN_FN) is #included via
// -DGEN_HDR; STATIC_FN / DYN_FN / ARGVAL come from -D so this one file serves
// both DMA paths and any future size.
//
//===----------------------------------------------------------------------===//

#include GEN_HDR

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

// Read a hex-per-line word stream (the "%08X\n" format emitted by
// aie-translate --aie-npu-to-binary -aie-output-binary=false).
std::vector<uint32_t> readHex(const char *path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open golden hex file '%s'\n", path);
    std::exit(2);
  }
  std::vector<uint32_t> words;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    words.push_back(
        static_cast<uint32_t>(std::strtoul(line.c_str(), nullptr, 16)));
  }
  return words;
}

// Replay a TXN word stream into the final register state it programs: a map
// from register address to the last value written there. A blockwrite of N
// payload words is N sequential word-writes (addr, addr+4, ...); a write32 is a
// single word-write. maskwrite/sync/address_patch are recorded positionally at
// synthetic high keys so a structural difference (missing/extra/reordered op)
// still surfaces, without needing to model their register semantics.
//
// This is the equivalence relation for the DYNAMIC size path: a runtime BD
// lowers to a zero-template blockwrite PLUS write32 overrides, whereas the
// static path bakes the values into the blockwrite. The two are NOT byte-equal,
// but they program the SAME final registers -- which is exactly what replay
// compares. (The static-vs-golden pair stays byte-exact via equal().)
using RegMap = std::map<uint64_t, uint32_t>;

bool replayToRegisters(const char *name, const std::vector<uint32_t> &txn,
                       RegMap &regs) {
  using namespace aie_runtime;
  if (txn.size() < 4) {
    std::fprintf(stderr, "%s: stream too short for header\n", name);
    return false;
  }
  size_t pos = 4;                 // skip the 4-word header
  uint64_t synthKey = 1ULL << 40; // positional keys for non-write ops
  while (pos < txn.size()) {
    uint32_t opc = txn[pos];
    switch (opc) {
    case TXN_OPC_WRITE: {
      if (pos + 6 > txn.size())
        break;
      regs[txn[pos + 2]] = txn[pos + 4];
      pos += 6;
      break;
    }
    case TXN_OPC_BLOCKWRITE: {
      if (pos + 4 > txn.size())
        break;
      uint32_t addr = txn[pos + 2];
      uint32_t byteSize = txn[pos + 3];
      size_t total = byteSize / sizeof(uint32_t);
      for (size_t i = 4; i < total; ++i)
        regs[addr + (i - 4) * 4] = txn[pos + i];
      pos += total;
      break;
    }
    case TXN_OPC_MASKWRITE: {
      // Apply as a masked RMW so it composes with any write to the same reg.
      if (pos + 7 > txn.size())
        break;
      uint32_t addr = txn[pos + 2], val = txn[pos + 4], mask = txn[pos + 5];
      regs[addr] = (regs[addr] & ~mask) | (val & mask);
      pos += 7;
      break;
    }
    case TXN_OPC_TCT: { // sync
      regs[synthKey++] = txn[pos + 2];
      regs[synthKey++] = txn[pos + 3];
      pos += 4;
      break;
    }
    case TXN_OPC_DDR_PATCH: {           // address_patch
      regs[synthKey++] = txn[pos + 6];  // register to patch
      regs[synthKey++] = txn[pos + 8];  // arg_idx
      regs[synthKey++] = txn[pos + 10]; // arg_plus
      pos += 12;
      break;
    }
    default:
      std::fprintf(stderr, "%s: unhandled TXN opcode 0x%x at word %zu\n", name,
                   opc, pos);
      return false;
    }
  }
  return true;
}

bool registersEqual(const char *aName, const std::vector<uint32_t> &a,
                    const char *bName, const std::vector<uint32_t> &b) {
  RegMap ra, rb;
  if (!replayToRegisters(aName, a, ra) || !replayToRegisters(bName, b, rb))
    return false;
  if (ra != rb) {
    std::fprintf(stderr, "%s and %s program different register state\n", aName,
                 bName);
    // Report the first differing key.
    auto ia = ra.begin(), ib = rb.begin();
    for (; ia != ra.end() && ib != rb.end(); ++ia, ++ib) {
      if (*ia != *ib) {
        std::fprintf(stderr, "  first diff at reg 0x%llx: 0x%08x vs 0x%08x\n",
                     (unsigned long long)ia->first, ia->second, ib->second);
        return false;
      }
    }
    return false;
  }
  return true;
}

// Report the first divergence (or a length mismatch) and return false.
bool equal(const char *aName, const std::vector<uint32_t> &a, const char *bName,
           const std::vector<uint32_t> &b) {
  size_t lim = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < lim; ++i) {
    if (a[i] != b[i]) {
      std::fprintf(
          stderr, "%s vs %s differ at word %zu: 0x%08llx vs 0x%08llx\n", aName,
          bName, i, (unsigned long long)a[i], (unsigned long long)b[i]);
      return false;
    }
  }
  if (a.size() != b.size()) {
    std::fprintf(stderr,
                 "%s (%zu words) and %s (%zu words) have equal prefix "
                 "but different length\n",
                 aName, a.size(), bName, b.size());
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <golden.hex>\n", argv[0]);
    return 2;
  }

  // The generated builders return std::optional: std::nullopt if a runtime
  // scalar would overflow a narrow BD field. For the equivalence check the
  // values are in range, so a nullopt is a harness failure.
  std::vector<uint32_t> golden = readHex(argv[1]);
  auto statOpt = STATIC_FN();
  auto dynOpt = DYN_FN(ARGVAL);
  if (!statOpt) {
    std::fprintf(stderr, "static builder returned nullopt (unexpected)\n");
    return 1;
  }
  if (!dynOpt) {
    std::fprintf(stderr, "dynamic builder returned nullopt for arg=%d\n",
                 (int)(ARGVAL));
    return 1;
  }
  const std::vector<uint32_t> &stat = *statOpt;
  const std::vector<uint32_t> &dyn = *dynOpt;

  // golden-vs-static is always byte-exact (both bake constants the same way).
  bool ok = equal("golden", golden, "static-C++", stat);

  // static-vs-dynamic: a runtime DMA *size* lowers to a zero-template
  // blockwrite + write32 overrides, which is not byte-equal to the static
  // baked blockwrite but programs identical registers. Compile the runtime-size
  // tests with -DDYN_STRUCTURAL to compare register state; the rtp-only tests
  // (whose dynamic stream stays byte-identical) keep the strict byte check.
#ifdef DYN_STRUCTURAL
  ok = ok && registersEqual("static-C++", stat, "dynamic-C++", dyn);
#else
  ok = ok && equal("static-C++", stat, "dynamic-C++", dyn);
#endif

  if (!ok)
    return 1;

  std::printf("equivalent: %zu words (arg=%d)\n", golden.size(), (int)(ARGVAL));
  return 0;
}
