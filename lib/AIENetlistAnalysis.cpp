// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "AIEDialect.h"
#include "AIENetlistAnalysis.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

void xilinx::AIE::NetlistAnalysis::runAnalysis() {
  
  // Collect op instances
  collectTiles(tiles);
  collectCores(cores);
  collectMems(mems);
  collectLocks(locks);
  collectBuffers(buffers);
  collectSwitchboxes(switchboxes);
}

void xilinx::AIE::NetlistAnalysis::collectTiles(DenseMap<std::pair<int, int>, Operation *> &tiles) {
  for (auto tile : module.getOps<TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[std::make_pair(colIndex, rowIndex)] = tile;
  }
}

void xilinx::AIE::NetlistAnalysis::collectCores(DenseMap<Operation *, CoreOp> &cores) {
  for (auto core : module.getOps<CoreOp>()) {
    Operation *tileOp = core.tile().getDefiningOp();
    assert(cores.count(tileOp) == 0 && "Invalid netlist! Expected 1-1 mapping of tile and core");
    cores[tileOp] = core;
  }
}

void xilinx::AIE::NetlistAnalysis::collectMems(DenseMap<Operation *, MemOp> &mems) {
  for (auto mem : module.getOps<MemOp>()) {
    Operation *tileOp = mem.tile().getDefiningOp();
    assert(mems.count(tileOp) == 0 && "Invalid netlist! Expected 1-1 mapping of tile and mem");
    mems[tileOp] = mem;
  }
}

void xilinx::AIE::NetlistAnalysis::collectLocks(DenseMap<std::pair<Operation *, int>, LockOp> &locks) {
  for (auto lock : module.getOps<LockOp>()) {
    Operation *tileOp = lock.tile().getDefiningOp();
    int lockID = lock.getLockID();
    assert(locks.count(std::make_pair(tileOp, lockID)) == 0 &&
           "Invalid netlist! Expected 1-1 mapping of (tile, lockID) and lock");
    locks[std::make_pair(tileOp, lockID)] = lock;
  }
}

void xilinx::AIE::NetlistAnalysis::collectBuffers(DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers) {
  for (auto buffer : module.getOps<BufferOp>()) {
    Operation *tileOp = buffer.tile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

void xilinx::AIE::NetlistAnalysis::collectSwitchboxes(DenseMap<Operation *, SwitchboxOp> &switchboxes) {
  for (auto switchbox : module.getOps<SwitchboxOp>()) {
    //Operation *tileOp = switchbox.tile().getDefiningOp();
  }
}

std::pair<int, int> xilinx::AIE::NetlistAnalysis::getCoord(Operation *Op) const {
  if (TileOp op = dyn_cast<TileOp>(Op))
    return std::make_pair(op.colIndex(), op.rowIndex());

  if (CoreOp op = dyn_cast<CoreOp>(Op))
    return std::make_pair(op.colIndex(), op.rowIndex());

  if (MemOp op = dyn_cast<MemOp>(Op))
    return std::make_pair(op.colIndex(), op.rowIndex());

  if (LockOp op = dyn_cast<LockOp>(Op)) {
    TileOp tile = dyn_cast<TileOp>(op.tile().getDefiningOp());
    return std::make_pair(tile.colIndex(), tile.rowIndex());
  }

  if (BufferOp op = dyn_cast<BufferOp>(Op)) {
    TileOp tile = dyn_cast<TileOp>(op.tile().getDefiningOp());
    return std::make_pair(tile.colIndex(), tile.rowIndex());
  }

  llvm_unreachable("Unknown Operation!");
}

// item: Lock, Buffer
// user: Core, Mem
bool xilinx::AIE::NetlistAnalysis::isLegalAffinity(Operation *item, Operation *user) const {
  std::pair<int, int> itemCoord = getCoord(item);
  std::pair<int, int> userCoord = getCoord(user);
  int itemCol = itemCoord.first;
  int itemRow = itemCoord.second;
  int userCol = userCoord.first;
  int userRow = userCoord.second;
  bool IsUserMem = isa<MemOp>(user);

  if (IsUserMem)
    return (isItself(itemCol, itemRow, userCol, userRow));

  // user is a Core
  return isLegalMemAffinity(userCol, userRow, itemCol, itemRow);
}

ArrayRef<Operation *> xilinx::AIE::NetlistAnalysis::getCompatibleTiles(Operation *Op) const {
  SmallVector<Operation *, 4> compTiles;
  // TODO
  return compTiles;
}

bool xilinx::AIE::NetlistAnalysis::validateCoreOrMemRegion(Operation *CoreOrMemOp) const {
  Region *r = nullptr;
  if (CoreOp core = dyn_cast<CoreOp>(CoreOrMemOp))
    r = &core.body();
  else if (MemOp mem = dyn_cast<MemOp>(CoreOrMemOp))
    r = &mem.body();

  assert(r && "Expected non-null region!");

  bool IsValid = true;
  bool IsCore = isa<CoreOp>(CoreOrMemOp);

  r->walk([&](Operation *Op) {
    // Check illegal uses of some Ops
    if (LockOp lock = dyn_cast<LockOp>(Op)) {
      llvm::dbgs() << "Invalid LockOp found in region: " << lock << '\n';
      IsValid = false;
    } else if (BufferOp buf = dyn_cast<BufferOp>(Op)) {
      llvm::dbgs() << "Invalid BufferOp found in region: " << buf << '\n';
      IsValid = false;
    } else if (DMAStartOp dma = dyn_cast<DMAStartOp>(Op)) {
      if (IsCore) {
        llvm::dbgs() << "Invalid DMAStartOp found in region: " << dma << '\n';
        IsValid = false;
      }
    } else if (DMABDOp dmaBd = dyn_cast<DMABDOp>(Op)) {
      if (IsCore) {
        llvm::dbgs() << "Invalid DMABDOp found in region: " << dmaBd << '\n';
        IsValid = false;
      }
    }

    // Check Op's operands
    for (Value operand : Op->getOperands()) {
      if (LockOp lock = dyn_cast<LockOp>(operand.getDefiningOp())) {
        IsValid = isLegalAffinity(lock, CoreOrMemOp);
        if (!IsValid)
          llvm::dbgs() << "Illegal use of lock in region: " << lock << '\n';
      } else if (BufferOp buf = dyn_cast<BufferOp>(operand.getDefiningOp())) {
        IsValid = isLegalAffinity(buf, CoreOrMemOp);
        if (!IsValid)
          llvm::dbgs() << "Illegal use of buffer in region: " << buf << '\n';
      } else {
        // except for (legal) lock and buffer, operand should be defined within the region
        IsValid = CoreOrMemOp->isProperAncestor(operand.getDefiningOp());
      }
    }

  });

  return IsValid;
}

int xilinx::AIE::NetlistAnalysis::getMemUsageInBytes(Operation *tileOp) const {
  int memUsage = 0;
  for (auto buf : buffers[tileOp]) {
    MemRefType t = buf.getType().cast<MemRefType>();
    memUsage += t.getSizeInBits();
  }
  return memUsage / 8;
}

int xilinx::AIE::NetlistAnalysis::getBufferBaseAddress(Operation *bufOp) const {
  BufferOp buf = dyn_cast<BufferOp>(bufOp);
  Operation *tileOp = buf.tile().getDefiningOp();
  int baseAddr = 0;
  for (unsigned i = 0; i < buffers[tileOp].size(); i++) {
    MemRefType t = buffers[tileOp][i].getType().cast<MemRefType>();
    if (bufOp == buffers[tileOp][i])
      break;
    baseAddr += (t.getSizeInBits() / 8);
  }

  return baseAddr;
}

void xilinx::AIE::NetlistAnalysis::print(raw_ostream &os) {
  // TODO
}
