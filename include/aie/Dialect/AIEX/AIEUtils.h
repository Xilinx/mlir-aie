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

// Find (or create) a private constant `memref.global` in `dev` holding `words`.
//
// `dedupCache`/`nextId`, when provided, make repeated calls O(1): without them,
// each call rescans every existing global for dedup and probes the symbol table
// for a free name, which is O(n) per call. Callers lowering many BDs (e.g. NPU
// DMA lowering of a large multi-column program) should pass both, seeded once
// from the device's existing globals. `dedupCache` maps each initial-value to
// its global; `*nextId` is the next free "blockwrite_data_<n>" index (one past
// the max already present), which the helper post-increments per created
// global. Seeding past the max guarantees unique names by construction (no
// symbol-table probe). Both are scoped to one device's lowering and hold raw op
// handles; discard them afterward.
memref::GlobalOp getOrCreateDataMemref(
    OpBuilder &builder, AIE::DeviceOp dev, mlir::Location loc,
    ArrayRef<uint32_t> words,
    llvm::DenseMap<mlir::Attribute, memref::GlobalOp> *dedupCache = nullptr,
    unsigned *nextId = nullptr);

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
//   <name> <state_table_idx> <type> <kind>
//   ...
// where kind is "core" (shift-2 encoded) or "addr" (raw).
void emitScratchpadParamsFile(mlir::ModuleOp moduleOp, llvm::raw_ostream &os);
} // namespace AIEX
} // namespace xilinx
