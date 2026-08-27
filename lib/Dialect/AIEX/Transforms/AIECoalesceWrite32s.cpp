//===- AIECoalesceWrite32s.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merges runs of npu.write32 operations at contiguous addresses into a single
// npu.blockwrite.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECOALESCEWRITE32S
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// One 32-bit write, together with the operation that produced it.
struct WriteWord {
  uint32_t address;
  uint32_t value;
  // Set for an npu.maskwrite32, which the pass keeps as it is.
  std::optional<uint32_t> mask;
  Operation *op;

  bool operator<(const WriteWord &other) const {
    return address < other.address;
  }

  bool isMaskWrite() const { return mask.has_value(); }
};

// A run of operations that the pass may reorder among themselves. A write to a
// special register or any operation the pass does not model ends the slice.
struct WriteSlice {
  SmallVector<WriteWord> writes;

  bool isEmpty() const { return writes.empty(); }
};

struct AIECoalesceWrite32sPass
    : xilinx::AIEX::impl::AIECoalesceWrite32sBase<AIECoalesceWrite32sPass> {

  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();
    SmallVector<AIE::RuntimeSequenceOp> runtimeSeqs;
    deviceOp.walk(
        [&](AIE::RuntimeSequenceOp seqOp) { runtimeSeqs.push_back(seqOp); });
    for (auto seqOp : runtimeSeqs)
      coalesceWrite32sInSequence(seqOp);
  }

private:
  void coalesceWrite32sInSequence(AIE::RuntimeSequenceOp seqOp) {
    OpBuilder builder(seqOp.getContext());
    auto deviceOp = seqOp->getParentOfType<AIE::DeviceOp>();
    if (!deviceOp)
      return;
    const auto &tm = deviceOp.getTargetModel();

    for (Block &block : seqOp.getBody()) {
      SmallVector<WriteSlice> slices;
      WriteSlice currentSlice;

      auto endSlice = [&]() {
        if (!currentSlice.isEmpty())
          slices.push_back(std::move(currentSlice));
        currentSlice = WriteSlice();
      };

      for (Operation &op : block) {
        // A side-effect-free operation, such as the arith.constant that feeds a
        // write or the memref.get_global that feeds a blockwrite, orders
        // nothing by itself.
        if (isMemoryEffectFree(&op))
          continue;

        if (auto write32Op = dyn_cast<NpuWrite32Op>(&op)) {
          std::optional<uint32_t> addr = write32Op.getAbsoluteAddress();
          std::optional<int64_t> value =
              getConstantIntValue(write32Op.getValue());
          if (!addr || !value || tm.isSpecialRegister(*addr)) {
            endSlice();
            continue;
          }
          currentSlice.writes.push_back(WriteWord{
              *addr, static_cast<uint32_t>(*value), std::nullopt, &op});
          continue;
        }

        if (auto maskWriteOp = dyn_cast<NpuMaskWrite32Op>(&op)) {
          std::optional<uint32_t> addr = maskWriteOp.getAbsoluteAddress();
          std::optional<int64_t> value =
              getConstantIntValue(maskWriteOp.getValue());
          std::optional<int64_t> mask =
              getConstantIntValue(maskWriteOp.getMask());
          if (!addr || !value || !mask || tm.isSpecialRegister(*addr)) {
            endSlice();
            continue;
          }
          currentSlice.writes.push_back(
              WriteWord{*addr, static_cast<uint32_t>(*value),
                        static_cast<uint32_t>(*mask), &op});
          continue;
        }

        if (auto blockWriteOp = dyn_cast<NpuBlockWriteOp>(&op)) {
          std::optional<uint32_t> addr = blockWriteOp.getAbsoluteAddress();
          DenseIntElementsAttr dataWords = blockWriteOp.getDataWords();
          if (!addr || !dataWords) {
            endSlice();
            continue;
          }
          bool touchesSpecialRegister = false;
          for (int64_t i = 0, e = dataWords.size(); i < e; ++i)
            if (tm.isSpecialRegister(*addr + i * 4)) {
              touchesSpecialRegister = true;
              break;
            }
          if (touchesSpecialRegister) {
            endSlice();
            continue;
          }
          int64_t idx = 0;
          for (auto val : dataWords.getValues<APInt>()) {
            currentSlice.writes.push_back(WriteWord{
                static_cast<uint32_t>(*addr + idx * 4),
                static_cast<uint32_t>(val.getZExtValue()), std::nullopt, &op});
            ++idx;
          }
          continue;
        }

        endSlice();
      }

      endSlice();

      for (auto &slice : slices)
        processSlice(builder, slice, deviceOp);
    }
  }

  void processSlice(OpBuilder &builder, WriteSlice &slice,
                    AIE::DeviceOp deviceOp) {
    if (slice.writes.empty())
      return;

    // Within a slice the last write to an address wins.
    DenseMap<uint32_t, size_t> lastWriteIndex;
    DenseSet<Operation *> supersededOps;
    for (size_t i = 0; i < slice.writes.size(); ++i) {
      uint32_t addr = slice.writes[i].address;
      auto it = lastWriteIndex.find(addr);
      if (it != lastWriteIndex.end())
        supersededOps.insert(slice.writes[it->second].op);
      lastWriteIndex[addr] = i;
    }

    SmallVector<WriteWord> uniqueWrites;
    for (size_t i = 0; i < slice.writes.size(); ++i)
      if (lastWriteIndex[slice.writes[i].address] == i)
        uniqueWrites.push_back(slice.writes[i]);
    slice.writes = std::move(uniqueWrites);

    // A blockwrite covers one address range, so sorting by address exposes the
    // runs the pass can merge.
    llvm::sort(slice.writes);

    SmallVector<SmallVector<WriteWord>> sequences;
    SmallVector<WriteWord> currentSeq;
    auto endSequence = [&]() {
      if (currentSeq.size() >= minWritesToCoalesce)
        sequences.push_back(currentSeq);
      currentSeq.clear();
    };

    for (auto &write : slice.writes) {
      // A masked write reads the register, so it stays a maskwrite32 and ends
      // the run.
      if (write.isMaskWrite()) {
        endSequence();
        continue;
      }
      if (!currentSeq.empty() && write.address != currentSeq.back().address + 4)
        endSequence();
      currentSeq.push_back(write);
    }
    endSequence();

    DenseSet<Operation *> toErase = supersededOps;
    for (auto &seq : sequences) {
      createBlockWrite(builder, seq, deviceOp);
      for (auto &write : seq)
        toErase.insert(write.op);
    }

    for (auto *op : toErase)
      op->erase();
  }

  void createBlockWrite(OpBuilder &builder, ArrayRef<WriteWord> sequence,
                        AIE::DeviceOp deviceOp) {
    if (sequence.empty())
      return;

    MLIRContext *ctx = builder.getContext();
    uint32_t startAddr = sequence.front().address;
    Location loc = sequence.front().op->getLoc();

    SmallVector<int32_t> values;
    for (auto &write : sequence)
      values.push_back(static_cast<int32_t>(write.value));

    auto i32Type = IntegerType::get(ctx, 32);
    auto shape = static_cast<int64_t>(values.size());
    auto memrefType = MemRefType::get({shape}, i32Type);
    auto valuesAttr = DenseElementsAttr::get<int32_t>(
        RankedTensorType::get({shape}, i32Type), ArrayRef<int32_t>(values));

    std::string globalName = "coalesced_write32_" + std::to_string(startAddr) +
                             "_" + std::to_string(globalCounter++);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(deviceOp.getBody());
    memref::GlobalOp::create(
        builder, loc, globalName,
        /*sym_visibility=*/builder.getStringAttr("private"),
        /*type=*/memrefType,
        /*initial_value=*/valuesAttr,
        /*constant=*/true,
        /*alignment=*/nullptr);

    builder.setInsertionPoint(sequence.front().op);
    auto getGlobalOp =
        memref::GetGlobalOp::create(builder, loc, memrefType, globalName);
    NpuBlockWriteOp::create(builder, loc, startAddr, getGlobalOp.getResult(),
                            /*buffer=*/nullptr, /*column=*/nullptr,
                            /*row=*/nullptr);
  }

  unsigned globalCounter = 0;
};

} // namespace

namespace xilinx::AIEX {
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}
} // namespace xilinx::AIEX
