//===- AIESplitConfigureEntries.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IRON's `OperatorSequence` lowering fuses a decode into a single module whose
// host device carries ONE `aie.runtime_sequence` holding MANY
// `aiex.configure @opK { ... aiex.run @sequence(subviews) }` ops in schedule
// order (llama: 322 configures over 19 distinct config devices, e.g. op0 x33).
// The union `--reconfig-method` flow downstream -- `applyReconfigMethod`
// (tools/aiecc) and the `aie-expand-load-pdi` self-clear -- instead assumes the
// host holds exactly ONE `aiex.configure`/`load_pdi` per runtime-sequence block
// (the `main:init`+`main:config_k` shape produced from N separate designs).
//
// This pass EXPLODES the single multi-configure sequence into N per-config
// runtime_sequences (one `aiex.configure` each), in original program order, so
// all the existing downstream union machinery is reused unchanged.
//
// Only the `aiex.entrypoint`-marked host device is touched; the config-template
// devices (`aie.device @opK`) and their offset-agnostic bodies are left alone.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESPLITCONFIGUREENTRIES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-split-configure-entries"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;
using namespace xilinx::AIE;

namespace {

// The entry-device marker `xilinx::AIEX::kEntrypointAttr` (AIEXDialect.h, in
// scope via `using namespace xilinx::AIEX`) is the sole discriminator that
// identifies the reconfiguration entry device; config-order / name suffixes and
// tile-lessness are all fragile. The `--reconfig-method` fold stamps it and
// readers match on `hasAttr(kEntrypointAttr)`.

// Split one multi-configure runtime sequence into N per-configure sequences.
// Returns failure (with a diagnostic already emitted) on a malformed input.
static LogicalResult splitOneSequence(RuntimeSequenceOp seq,
                                      llvm::StringSet<> &seenNames) {
  Block &body = seq.getBody().front();

  // Top-level configures in block (= schedule) order. Only direct children are
  // considered; a configure nested inside control flow is not the IRON shape.
  llvm::SmallVector<AIEX::ConfigureOp> configures;
  for (AIEX::ConfigureOp cfg : body.getOps<AIEX::ConfigureOp>())
    configures.push_back(cfg);

  // Nothing to explode. Leave a sequence that is not a multi-configure host
  // sequence untouched (empty, or already one-configure-per-sequence): the
  // downstream machinery already accepts that shape.
  if (configures.size() <= 1)
    return success();

  // Any op other than the configures themselves in the top-level block would be
  // silently dropped by the per-configure move below (each new sequence holds
  // exactly its one configure). The IRON monolith is nothing but configures, so
  // fail loudly rather than lose an instruction.
  for (Operation &op : body)
    if (!llvm::isa<AIEX::ConfigureOp>(op)) {
      op.emitError("aie-split-configure-entries: the marked host runtime "
                   "sequence must contain only aiex.configure ops; found '")
          << op.getName() << "'";
      return failure();
    }

  // Signature shared by every exploded sequence: the subviews inside each
  // configure re-reference these arguments (remapped per new sequence below).
  auto argTypes = body.getArgumentTypes();
  llvm::SmallVector<Location> argLocs;
  for (BlockArgument arg : body.getArguments())
    argLocs.push_back(arg.getLoc());

  OpBuilder builder(seq);
  for (auto [index, cfg] : llvm::enumerate(configures)) {
    // Each configure must be self-contained: everything it references from
    // above must be a block argument of the original sequence (the subviews it
    // owns slice those args). A capture of some other value defined in the host
    // block would dangle once the configure is cloned into a fresh sequence --
    // fail loudly rather than emit invalid IR.
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(cfg.getBody(), captured);
    for (Value v : captured) {
      auto blockArg = llvm::dyn_cast<BlockArgument>(v);
      if (!blockArg || blockArg.getOwner() != &body) {
        cfg.emitError("aie-split-configure-entries: aiex.configure captures a "
                      "value that is not a host runtime-sequence argument; "
                      "cannot split it into its own sequence");
        return failure();
      }
    }

    // `<referenced-device-symbol>_<block-order-index>`: the index disambiguates
    // repeated occurrences of the same device and records schedule position.
    std::string name = (cfg.getSymbol() + "_" + llvm::Twine(index)).str();
    if (!seenNames.insert(name).second) {
      // Cannot happen with the block-order index (it is unique per sequence),
      // but mirror applyReconfigMethod's loud duplicate-name guard.
      cfg.emitError("aie-split-configure-entries: duplicate entry sequence "
                    "name '")
          << name << "'";
      return failure();
    }

    // Fresh runtime sequence with the shared signature, inserted before the
    // original so program order is preserved as new sequences accumulate.
    builder.setInsertionPoint(seq);
    auto newSeq = RuntimeSequenceOp::create(builder, cfg.getLoc(),
                                            /*sym_name=*/StringAttr{},
                                            /*emit_parameter_sync_preamble=*/
                                            BoolAttr{});
    newSeq.setSymName(name);
    Block *newBlock = new Block;
    newSeq.getBody().push_back(newBlock);
    newBlock->addArguments(argTypes, argLocs);

    // Clone the configure into the new sequence, remapping the host block args
    // to the new block's args (index-for-index). Cloning walks the nested
    // region -- the owned subviews and the run pick up the new args for free.
    IRMapping map;
    for (auto [oldArg, newArg] :
         llvm::zip(body.getArguments(), newBlock->getArguments()))
      map.map(oldArg, newArg);
    builder.setInsertionPointToEnd(newBlock);
    builder.clone(*cfg.getOperation(), map);
  }

  // The original multi-configure sequence has been fully re-expressed.
  seq.erase();
  return success();
}

struct AIESplitConfigureEntriesPass
    : public xilinx::AIEX::impl::AIESplitConfigureEntriesBase<
          AIESplitConfigureEntriesPass> {
  using AIESplitConfigureEntriesBase::AIESplitConfigureEntriesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (DeviceOp dev : module.getOps<DeviceOp>()) {
      if (!dev->hasAttr(kEntrypointAttr))
        continue;

      // Names are unique across the whole marked device (the dispatch name is
      // the sym_name), so guard collisions across every exploded sequence.
      llvm::StringSet<> seenNames;

      // Snapshot the sequences first: splitOneSequence inserts new sequences
      // and erases the original, so iterating the live op list would revisit
      // freshly-created (single-configure) sequences.
      llvm::SmallVector<RuntimeSequenceOp> seqs(
          dev.getOps<RuntimeSequenceOp>());
      for (RuntimeSequenceOp seq : seqs) {
        if (seq.getBody().empty())
          continue;
        if (failed(splitOneSequence(seq, seenNames))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIESplitConfigureEntriesPass() {
  return std::make_unique<AIESplitConfigureEntriesPass>();
}
