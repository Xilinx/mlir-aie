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

void xilinx::AIE::NetlistAnalysis::collectBuffers(
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers) {

  for (auto buffer : module.getOps<BufferOp>()) {
    Operation *tileOp = buffer.tile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

void xilinx::AIE::NetlistAnalysis::collectSwitchboxes(
  DenseMap<Operation *, SwitchboxOp> &switchboxes) {

  for (auto switchbox : module.getOps<SwitchboxOp>()) {
    Operation *tileOp = switchbox.tile().getDefiningOp();
    assert(switchboxes.count(tileOp) == 0 && "Invalid netlist! Expected 1-1 mapping of tile and switchbox");
    switchboxes[tileOp] = switchbox;
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
    return (isInternal(itemCol, itemRow, userCol, userRow));

  // user is a Core
  return isLegalMemAffinity(userCol, userRow, itemCol, itemRow);
}

bool xilinx::AIE::NetlistAnalysis::validateCoreOrMemRegion(Operation *CoreOrMemOp) {
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
      assert(false && "Invalid LockOp found in region");
    } else if (BufferOp buf = dyn_cast<BufferOp>(Op)) {
      assert(false && "Invalid BufferOp found in region");
    } else if (DMAStartOp dma = dyn_cast<DMAStartOp>(Op)) {
      if (IsCore) {
        assert(false && "Invalid DMAStartOp found in region");
      }
    } else if (DMABDOp dmaBd = dyn_cast<DMABDOp>(Op)) {
      if (IsCore) {
        assert(false && "Invalid DMABDOp found in region");
      }
    }

    // Check Op's operands
    for (Value operand : Op->getOperands()) {
      if (LockOp lock = dyn_cast<LockOp>(operand.getDefiningOp())) {
        IsValid = isLegalAffinity(lock, CoreOrMemOp);
        assert(IsValid && "Illegal use of lock in region");
      } else if (BufferOp buf = dyn_cast<BufferOp>(operand.getDefiningOp())) {
        IsValid = isLegalAffinity(buf, CoreOrMemOp);
        assert(IsValid && "Illegal use of buffer in region");
        bufferUsers[buf].push_back(CoreOrMemOp);
      } else {
        // except for (legal) lock and buffer, operand should be defined within the region
        IsValid = CoreOrMemOp->isProperAncestor(operand.getDefiningOp());
      }
    }

  });

  return IsValid;
}

void xilinx::AIE::NetlistAnalysis::collectBufferUsage() {
  SmallVector<Operation *, 4> CoreOrMemOps;
  for (auto map : cores) {
    CoreOrMemOps.push_back(map.second);
  }

  for (auto map : mems) {
    CoreOrMemOps.push_back(map.second);
  }

  for (auto CoreOrMemOp : CoreOrMemOps) {
    Region *r = nullptr;
    if (CoreOp core = dyn_cast<CoreOp>(CoreOrMemOp))
      r = &core.body();
    else if (MemOp mem = dyn_cast<MemOp>(CoreOrMemOp))
      r = &mem.body();

    assert(r && "Expected non-null region!");

    r->walk([&](Operation *Op) {
      for (Value operand : Op->getOperands()) {
        if (BufferOp buf = dyn_cast<BufferOp>(operand.getDefiningOp())) {
          bufferUsers[buf].push_back(CoreOrMemOp);
        }
      }
    });
  }
}

void xilinx::AIE::NetlistAnalysis::collectDMAUsage() {
  for (auto map : mems) {
    MemOp mem = map.second;
    Region &r = mem.body();
    Block *endBlock = &r.back();
    for (auto op : r.getOps<CondBranchOp>()) {
      DMAStartOp dmaSt = dyn_cast<DMAStartOp>(op.getCondition().getDefiningOp());
      int channelNum = dmaSt.getChannelNum();
      dmas[std::make_pair(mem, channelNum)] = dmaSt;
      Block *firstBd = op.getTrueDest();
      Block *curBd = firstBd;

      while (curBd != endBlock) {
        for (auto bdOp : curBd->getOps<DMABDOp>()) {
          Operation *buf = bdOp.buffer().getDefiningOp();
          if (std::find(dma2BufMap[dmaSt].begin(),
                        dma2BufMap[dmaSt].end(), buf) != dma2BufMap[dmaSt].end())
            continue;

          dma2BufMap[dmaSt].push_back(bdOp.buffer().getDefiningOp());
        }
        curBd = curBd->getSuccessors()[0];
      }
    }
  }
}

uint64_t xilinx::AIE::NetlistAnalysis::getMemUsageInBytes(Operation *tileOp) const {
  uint64_t memUsage = 0;
  for (auto buf : buffers[tileOp]) {
    MemRefType t = buf.getType().cast<MemRefType>();
    memUsage += t.getSizeInBits();
  }
  return memUsage / 8;
}

// FIXME: make address assignment for buffers explicit and move this function to
// an interface
uint64_t xilinx::AIE::NetlistAnalysis::getBufferBaseAddress(Operation *bufOp) const {
  if (auto buf = dyn_cast<BufferOp>(bufOp)) {
    return buf.address();
  } else if (auto buf = dyn_cast<ExternalBufferOp>(bufOp)) {
    return buf.address();
  } else {
    llvm_unreachable("unknown buffer type");
  }
}

SmallVector<Operation *, 4> xilinx::AIE::NetlistAnalysis::getNextConnectOps(
  ConnectOp currentConnect) const {

  SmallVector<Operation *, 4> nextConnectOps;

  Operation *swboxOp = currentConnect->getParentOp();
  SwitchboxOp swbox = dyn_cast<SwitchboxOp>(swboxOp);
  int col = swbox.colIndex();
  int row = swbox.rowIndex();
  int nextCol = -1;
  int nextRow = -1;
  WireBundle nextSrcBundle;
  int nextSrcIndex = currentConnect.destIndex();

  if (currentConnect.destBundle() == WireBundle::South) {
    nextCol = col;
    nextRow = row - 1;
    nextSrcBundle = WireBundle::North;
  } else if (currentConnect.destBundle() == WireBundle::West) {
    nextCol = col - 1;
    nextRow = row;
    nextSrcBundle = WireBundle::East;
  } else if (currentConnect.destBundle() == WireBundle::North) {
    nextCol = col;
    nextRow = row + 1;
    nextSrcBundle = WireBundle::South;
  } else if (currentConnect.destBundle() == WireBundle::East) {
    nextCol = col + 1;
    nextRow = row;
    nextSrcBundle = WireBundle::West;
  } else {
    return nextConnectOps;
  }

  assert((nextCol >= 0 && nextRow >= 0) && "Invalid ConnectOp! Could not find next tile!");

  Operation *nextTileOp = tiles[std::make_pair(nextCol, nextRow)];
  Operation *nextSwboxOp = switchboxes[nextTileOp];
  SwitchboxOp nextSwbox = dyn_cast<SwitchboxOp>(nextSwboxOp);

  for (auto connect : nextSwbox.getOps<ConnectOp>()) {
    if (connect.sourceBundle() == nextSrcBundle && connect.sourceIndex() == nextSrcIndex) {
      nextConnectOps.push_back(connect);
    }
  }

  return nextConnectOps;
}

SmallVector<Operation *, 4> xilinx::AIE::NetlistAnalysis::findRoutes(
  Operation *sourceConnectOp, Operation *destConnectOp) const {

  SmallVector<Operation *, 4> routes;
  routes.push_back(sourceConnectOp);
  ConnectOp sourceConnect = dyn_cast<ConnectOp>(sourceConnectOp);
  ArrayRef<Operation *> nextConnectOps(getNextConnectOps(sourceConnect));
  for (auto nextConnectOp : nextConnectOps) {
    if (destConnectOp == nextConnectOp) {
      return routes;
    } else {
      SmallVector<Operation *, 4> workList(findRoutes(nextConnectOp, destConnectOp));
      if (workList.size() > 0) {
        routes.push_back(destConnectOp);
        routes.insert(routes.end(), workList.begin(), workList.end());
        return routes;
      }
    }
  }

  return routes;
}

SmallVector<Operation *, 4> xilinx::AIE::NetlistAnalysis::findDestConnectOps(
  ConnectOp source, WireBundle destBundle) const {

  SmallVector<Operation *, 4> dests;
  SmallVector<Operation *, 4> workList;
  workList.push_back(source);

  while (!workList.empty()) {
    ConnectOp visitor = dyn_cast<ConnectOp>(workList.pop_back_val());
    ArrayRef<Operation *> nextConnectOps(getNextConnectOps(visitor));
    for (auto nextConnectOp : nextConnectOps) {
      ConnectOp nextConnect = dyn_cast<ConnectOp>(nextConnectOp);
      if (nextConnect.destBundle() != destBundle)
        workList.push_back(nextConnect);
      else
        dests.push_back(nextConnect);

    }
  }
  
  return dests;
}

void xilinx::AIE::NetlistAnalysis::dmaAnalysis() {
  // Source(DMAChannel, <Buffer0, Buffer1, ...>) --> Dest(DMAChannel, <Buffer0, Buffer1, ...>)
  collectDMAUsage();

  for (auto map : dma2BufMap) {
    Operation *srcDmaOp = map.first;
    DMAStartOp srcDma = dyn_cast<DMAStartOp>(srcDmaOp);
    if (srcDma.isRecv())
      continue;

    int srcChannelIndex = srcDma.getSendChannelIndex();

    Operation *srcMemOp = srcDmaOp->getParentOp();
    MemOp srcMem = dyn_cast<MemOp>(srcMemOp);
    SwitchboxOp swbox = switchboxes[srcMem.tile().getDefiningOp()];
    for (auto connect : swbox.getOps<ConnectOp>()) {
      WireBundle srcBundle = connect.sourceBundle();
      int srcIndex = connect.sourceIndex();
      if (!(srcBundle == WireBundle::DMA && srcIndex == srcChannelIndex))
        continue;

      dma2ConnectsMap[srcDmaOp].push_back(connect);

      ArrayRef<Operation *> destConnectOps(findDestConnectOps(connect, WireBundle::DMA));
      for (auto destConnectOp : destConnectOps) {
        ConnectOp destConnect = dyn_cast<ConnectOp>(destConnectOp);
        SwitchboxOp destSwbox = dyn_cast<SwitchboxOp>(destConnect->getParentOp());
        Operation *destMemOp = mems[destSwbox.tile().getDefiningOp()];
        int destChannelIndex = destConnect.destIndex();
        Operation *destDmaOp = dmas[std::make_pair(destMemOp, destChannelIndex)];
        dmaConnections[srcDma].push_back(destDmaOp);
        dma2ConnectsMap[destDmaOp].push_back(destConnect);
      }
    }
  }
}

void xilinx::AIE::NetlistAnalysis::lockAnalysis() {

  DenseMap<Value, SmallVector<Operation *, 4>> visitors;

  module.getBodyRegion().walk([&](Operation *Op) {
    if (auto op = dyn_cast<UseLockOp>(Op)) {
      Value lock = op.lock();
      if (op.acquire()) {
        visitors[lock].push_back(op);
      } else if (op.release()) {
        if (!visitors[lock].empty()) {
          Operation *Op = visitors[lock].pop_back_val();
          lockPairs[Op] = op;
        }
      }
    } else {
      for (Value operand : Op->getOperands()) {
        if (BufferOp buf = dyn_cast<BufferOp>(operand.getDefiningOp())) {
          for (auto map : visitors) {
            SmallVector<Operation *, 4> acqLocks(map.second);
            for (auto acqLock : acqLocks)
              bufAcqLocks[operand.getDefiningOp()].push_back(acqLock);
          }
        }
      }
    }
  });

  for (auto pair1 : lockPairs) {
    Operation *srcRelLockOp = pair1.second;
    for (auto pair2 :  lockPairs) {
      Operation *dstAcqLockOp = pair2.first;

      if (pair1 == pair2)
        continue;

      UseLockOp srcRelLock = dyn_cast<UseLockOp>(srcRelLockOp);
      UseLockOp dstAcqLock = dyn_cast<UseLockOp>(dstAcqLockOp);
      int relValue = srcRelLock.getLockValue();
      int acqValue = dstAcqLock.getLockValue();
      if (acqValue == 0)
        continue;

      if (relValue == acqValue)
        lockChains.push_back(std::make_pair(srcRelLockOp, dstAcqLockOp));
    }
  }
}

int xilinx::AIE::NetlistAnalysis::getAvailableLockID(Operation *tileOp) {
  for (unsigned i = 0; i < 16; i++) {
    if (locks.count(std::make_pair(tileOp, i)) == 0)
      return i;
  }

  return -1;
}

void xilinx::AIE::NetlistAnalysis::print(raw_ostream &os) {
  // TODO
}
