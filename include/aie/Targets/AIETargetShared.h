//===- AIETargetShared.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIETargetShared_XAIEV2_CDO_H
#define AIETargetShared_XAIEV2_CDO_H

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace xilinx {
namespace AIE {

std::string tileLocStr(llvm::StringRef col, llvm::StringRef row);

std::string tileLocStr(int col, int row);

std::string tileDMAInstStr(llvm::StringRef col, llvm::StringRef row,
                           llvm::StringRef bdNum);

std::string tileDMAInstStr(int col, int row, int bdNum);

std::string tileDMAInstRefStr(llvm::StringRef col, llvm::StringRef row,
                              llvm::StringRef bdNum);

std::string tileDMAInstRefStr(int col, int row, int bdNum);

std::string packetStr(llvm::StringRef id, llvm::StringRef type);

std::string packetStr(int id, int type);

void generateXAieDmaSetMultiDimAddr(llvm::raw_ostream &output, int ndims,
                                    llvm::ArrayRef<BDDimLayoutAttr> dims,
                                    int col, int row, int bdNum, int baseAddrA,
                                    int64_t offsetA, uint32_t lenA,
                                    int elementWidthInBytes,
                                    const char *errorRet);

llvm::SetVector<mlir::Block *> getOrderedChainOfBlocks(mlir::Region *region);

/// Collect every BD block reached from an out-of-order aie.dma_start channel.
/// These BDs use use_next_bd=0 (placement is by header id, not the chain).
llvm::SmallPtrSet<mlir::Block *, 8>
collectOutOfOrderBlocks(const llvm::SetVector<mlir::Block *> &blockVector);

/// A buffer's `initial_value` as the bytes to write to its tile memory, in
/// element order. Every path that delivers that value to the device calls this
/// -- the CDO and transaction writer, and the aiesim configuration source -- so
/// one buffer is initialized one way.
///
/// Returns nullopt for an element type with no byte image, meaning neither
/// integer nor float. The callers report that type.
std::optional<std::vector<char>>
denseAttrToBytes(mlir::DenseElementsAttr denseInit);

/// The extent the core compiler gets for its own .data, .rodata and .bss,
/// tile-relative. A `core_data` buffer among `buffers` fixes the extent;
/// otherwise it is the largest run the buffers leave above the stack.
///
/// The linker script emitter and the aiecc driver both need this, so a
/// diagnostic about the region names the bytes the script grants.
MemoryRun coreDataRegion(TileOp tile, llvm::ArrayRef<BufferOp> buffers);

} // namespace AIE
} // namespace xilinx

#endif
