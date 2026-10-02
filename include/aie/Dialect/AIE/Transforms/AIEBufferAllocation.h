//===- AIEBufferAllocation.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

namespace xilinx::AIE {

/// A memtile is reached by DMA rather than by core vector load and store, so
/// its bus width also covers the vector-alignment requirement.
struct TileMemoryLimits {
  int64_t maxDataMemorySize;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
};
inline TileMemoryLimits tileMemoryLimits(const AIETargetModel &targetModel,
                                         AIETileType tileType) {
  if (tileType == AIETileType::MemTile) {
    return {targetModel.getMemTileSize(),
            targetModel.getMemTileLoadStoreBusWidth(),
            targetModel.getMemTileLoadStoreBusWidth()};
  }
  return {targetModel.getLocalMemorySize(),
          targetModel.getComputeTileLoadStoreBusWidth(),
          targetModel.getComputeTileMaxVectorAlignBits()};
}

/// Return the alignment (in bits) a buffer of `sizeBytes` must satisfy.
///
/// Bus width alone is insufficient: from AIE2P on, a full-width vector access
/// needs 512-bit alignment while the bus is 256 bits wide. An externally
/// compiled kernel may perform such an access, so a buffer large enough to hold
/// a full-width vector gets the stricter alignment. A smaller buffer keeps the
/// bus width and costs no padding.
inline uint32_t requiredBufferAlignBits(uint32_t busAlignBits,
                                        uint32_t maxVecAlignBits,
                                        int64_t sizeBytes) {
  if (maxVecAlignBits <= busAlignBits)
    return busAlignBits;
  return sizeBytes * 8 >= static_cast<int64_t>(maxVecAlignBits)
             ? maxVecAlignBits
             : busAlignBits;
}

struct BufferAllocation {
  int64_t size;
  int64_t alignment;
  std::optional<int64_t> address;
};

/// The basic address allocator and resource planner share this layout: retain
/// address pins, then place unpinned buffers largest-first, aligning each start
/// and skipping occupied ranges. Return the high-water mark, without padding
/// after the final buffer. Entries retain their input order.
inline int64_t
assignSequentialBufferAddresses(llvm::MutableArrayRef<BufferAllocation> buffers,
                                int64_t start = 0) {
  llvm::SmallVector<std::pair<int64_t, int64_t>> pinned;
  llvm::SmallVector<BufferAllocation *> pending;
  int64_t highWater = start;
  for (auto &buffer : buffers) {
    if (buffer.address) {
      pinned.emplace_back(*buffer.address, buffer.size);
      highWater = std::max(highWater, *buffer.address + buffer.size);
    } else {
      pending.push_back(&buffer);
    }
  }
  llvm::stable_sort(pinned, [](auto a, auto b) { return a.first < b.first; });
  llvm::stable_sort(pending,
                    [](auto *a, auto *b) { return a->size > b->size; });
  auto *current = pinned.begin();
  int64_t address = start;
  for (auto *buffer : pending) {
    address = llvm::alignTo(address, buffer->alignment);
    while (current != pinned.end() && address + buffer->size > current->first) {
      address =
          llvm::alignTo(current->first + current->second, buffer->alignment);
      ++current;
    }
    buffer->address = address;
    address += buffer->size;
  }
  return std::max(highWater, address);
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEBUFFERALLOCATION_H
