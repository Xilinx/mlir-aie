// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include GEN_HDR

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

int main(int argc, char **argv) {
#ifdef SCALAR
  if (argc != 3)
    return 2;
  auto txn = GEN_FN(std::stoll(argv[2]));
#else
  if (argc != 2)
    return 2;
  auto txn = GEN_FN();
#endif
  if (!txn)
    return 3;
#ifdef CHECK_SHIM
  if (std::strcmp(dispatch_abi(), ABI_STRING) != 0)
    return 5;
  uint32_t *words = nullptr;
#ifdef SCALAR
  int64_t count = dispatch_generate(std::stoll(argv[2]), &words);
#else
  int64_t count = dispatch_generate(&words);
#endif
  if (count != static_cast<int64_t>(txn->size()) ||
      !std::equal(txn->begin(), txn->end(), words))
    return 6;
#endif
  std::ofstream out(argv[1], std::ios::binary);
  for (uint32_t word : *txn)
    for (unsigned byte = 0; byte != 4; ++byte)
      out.put(static_cast<char>(word >> (8 * byte)));
  out.close();
  return out ? 0 : 4;
}
