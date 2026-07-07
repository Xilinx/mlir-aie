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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

  bool ok = equal("golden", golden, "static-C++", stat) &&
            equal("static-C++", stat, "dynamic-C++", dyn);

  if (!ok)
    return 1;

  std::printf("equivalent: %zu words (arg=%d)\n", golden.size(), (int)(ARGVAL));
  return 0;
}
