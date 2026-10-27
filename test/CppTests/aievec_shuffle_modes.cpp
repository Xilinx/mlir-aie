//===- aievec_shuffle_modes.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks the byte permutation the compiler derives for every aievec.shuffle
// mode against aie2p_vshuffle_modes.json. That table was recorded on AIE2P
// (NPU2) hardware by running vshuffle in every mode on two 64-byte inputs
// holding the bytes 0 to 127; byte i of the result came from byte perm[i].
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecShuffleModes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace xilinx::aievec;

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: " << argv[0] << " aie2p_vshuffle_modes.json\n";
    return 2;
  }
  auto buffer = llvm::MemoryBuffer::getFile(argv[1]);
  if (!buffer) {
    llvm::errs() << "cannot read " << argv[1] << "\n";
    return 2;
  }
  llvm::Expected<llvm::json::Value> json =
      llvm::json::parse((*buffer)->getBuffer());
  if (!json) {
    llvm::errs() << llvm::toString(json.takeError()) << "\n";
    return 2;
  }
  const llvm::json::Object *modes =
      json->getAsObject() ? json->getAsObject()->getObject("modes") : nullptr;
  if (!modes) {
    llvm::errs() << "no \"modes\" object\n";
    return 2;
  }

  unsigned failures = 0, checked = 0;
  for (uint32_t m = 0; m <= getMaxEnumValForShuffleMode(); ++m) {
    std::optional<ShuffleMode> mode = symbolizeShuffleMode(m);
    if (!mode)
      continue;
    const llvm::json::Object *entry = modes->getObject(std::to_string(m));
    if (!entry) {
      llvm::errs() << "mode " << m << " (" << stringifyShuffleMode(*mode)
                   << ") is not in the hardware table\n";
      ++failures;
      continue;
    }
    std::optional<llvm::StringRef> name = entry->getString("name");
    if (!name || !name->equals_insensitive(stringifyShuffleMode(*mode))) {
      llvm::errs() << "mode " << m << " is " << stringifyShuffleMode(*mode)
                   << " here but " << name.value_or("?") << " on hardware\n";
      ++failures;
    }
    const llvm::json::Array *perm = entry->getArray("perm");
    std::array<uint8_t, 64> expected = getShuffleModeBytePermutation(*mode);
    if (!perm || perm->size() != expected.size()) {
      llvm::errs() << "mode " << m << ": no 64-byte permutation\n";
      ++failures;
      continue;
    }
    for (unsigned i = 0; i < expected.size(); ++i) {
      std::optional<int64_t> byte = (*perm)[i].getAsInteger();
      if (!byte || *byte != expected[i]) {
        llvm::errs() << "mode " << m << " (" << stringifyShuffleMode(*mode)
                     << "): result byte " << i << " comes from byte "
                     << byte.value_or(-1) << " on hardware, "
                     << static_cast<int>(expected[i]) << " here\n";
        ++failures;
        break;
      }
    }
    ++checked;
  }
  if (checked == 0) {
    llvm::errs() << "no mode checked\n";
    return 1;
  }
  llvm::outs() << checked << " modes checked, " << failures << " failures\n";
  return failures ? 1 : 0;
}
