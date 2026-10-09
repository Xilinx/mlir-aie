//===- AIEUtils.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace xilinx {
namespace AIEX {

// The private constant `memref.global`s holding a device's blockwrite payloads:
// one per distinct payload, each new one named `<prefix><n>` past every symbol
// of that form the device held when the table was made. It holds op handles,
// so it lives no longer than one rewrite of the device.
class BlockwriteData {
public:
  BlockwriteData(AIE::DeviceOp dev, llvm::StringRef prefix);

  // The global holding `words`, created at `builder`'s insertion point if the
  // device has none.
  memref::GlobalOp getOrCreate(OpBuilder &builder, mlir::Location loc,
                               ArrayRef<uint32_t> words);

  AIE::DeviceOp getDevice() const { return dev; }

private:
  AIE::DeviceOp dev;
  std::string prefix;
  llvm::DenseMap<mlir::Attribute, memref::GlobalOp> byValue;
  unsigned nextId = 0;
};

// Result of tracing through supported view/cast operations to a block argument
// for traceSubviewToBlockArgument function.
struct SubviewTraceResult {
  BlockArgument rootArg;
  int64_t offsetInBytes;
};

// Trace through memref.subview, memref.view, memref.cast, and
// memref.reinterpret_cast operations until the referenced SSA value is a block
// argument.
//
// Returns the root block argument and cumulative byte offset, or std::nullopt
// if the chain doesn't lead to a block argument or contains unsupported ops.
//
// This function checks that all subviews remain static and contiguous. A
// memref.view must have a constant byte shift and no dynamic result sizes.
std::optional<SubviewTraceResult> traceSubviewToBlockArgument(Value value);

// Index of `arg` among the host buffers of its runtime sequence.
//
// The host passes one buffer per memref argument. Scalar arguments travel in
// the instruction stream and occupy no buffer slot. The block-argument number
// therefore over-counts them. Returns nullopt when `arg` is not a memref.
std::optional<unsigned> getHostBufferArgIndex(BlockArgument arg);

// Emit an `aiex.npu.update_from_scratchpad` op that adds the runtime offset
// (held in the scratchpad slot referenced by `bdOp`'s
// `offset_state_table_idx` attribute, multiplied by the element size of
// `bufType`) into the BD address register at `registerAddr`.
LogicalResult emitUpdateBdAddressFromOffsetParameter(OpBuilder &builder,
                                                     Operation *bdOp,
                                                     BaseMemRefType bufType,
                                                     uint64_t registerAddr);

// Emit an `aiex.npu.update_from_scratchpad` op that adds the runtime length
// (held in the scratchpad slot referenced by `bdOp`'s
// `length_state_table_idx` attribute, times `lengthUnit` elements of
// `bufType`) into the buffer length register of BD `bdId` on tile
// (`col`, `row`), which the firmware counts in 32-bit words. Fails if the tile
// does not support a runtime length.
LogicalResult
emitUpdateBdLengthFromParameter(OpBuilder &builder, Operation *bdOp,
                                BaseMemRefType bufType, int64_t lengthUnit,
                                const AIE::AIETargetModel &targetModel, int col,
                                int row, int bdId);

// The configures a DMA task value can come from. A task carried through
// runtime control flow is an scf.for iter_arg or an scf.for/scf.if result
// rather than a configure result; this walks such a value back through every
// region-branch operand that can feed it (a loop's init and back-edge, each
// branch's yield), at any nesting depth. Each configure is listed once.
// Returns false if some path ends at a value that is not a configure result.
bool getReachableConfigures(Value task,
                            SmallVectorImpl<DMAConfigureTaskOp> &configures);

// The one configure a task value can come from, through runtime control flow
// as above, or null if there is none or more than one.
DMAConfigureTaskOp getUniqueReachableConfigure(Value task);

// Emit the params.txt description of every `aiex.scratchpad_parameter` in
// `moduleOp` (with their assigned `state_table_idx`/`kind`) to `os`.
//
// Format (one entry per line, easily parsed with std::ifstream >>):
//   <num_parameters>
//   <name> <state_table_idx> <type> <kind> <min> <max>
//   ...
// where kind is "core" (shift-2 encoded, for read_scratchpad_parameter) or
// "addr" (raw, for offset_parameter on DMA ops; also for a length_parameter
// with no core use), and <min> <max> is the range a DMA offset or length
// parameter must stay in, or "- -" for any other parameter.
void emitScratchpadParamsFile(mlir::ModuleOp moduleOp, llvm::raw_ostream &os);
} // namespace AIEX
} // namespace xilinx
