//===- AIETargetXAIEV2.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "aie/AIEDialect.h"
#include "aie/AIENetlistAnalysis.h"

#include "AIETargets.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx {
namespace AIE {

static std::string shimDMAInstStr(StringRef col, StringRef index) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "ShimDMAInst_" << col << "_" << index;
  return str;
}
static std::string tileLocStr(StringRef col, StringRef row) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "XAie_TileLoc(" << col << "," << row << ")";
  return str;
}
static std::string tileLocStr(int col, int row) {
  return tileLocStr(std::to_string(col), std::to_string(row));
}
static std::string tileDMAInstStr(StringRef col, StringRef row, 
                                  StringRef bdNum) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "dma_tile" << col << row << "_bd" << bdNum;
  return str;
}
static std::string tileDMAInstStr(int col, int row, int bdNum) {
  return tileDMAInstStr(std::to_string(col), std::to_string(row), 
                        std::to_string(bdNum));
}
static std::string tileDMAInstRefStr(StringRef col, StringRef row, 
                                     StringRef bdNum) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "&(" << tileDMAInstStr(col,row,bdNum) << ")";
  return str;
}
static std::string tileDMAInstRefStr(int col, int row, int bdNum) {
  return tileDMAInstRefStr(std::to_string(col), std::to_string(row), 
                           std::to_string(bdNum));
}
static std::string tileLockStr(StringRef id, StringRef val) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  //rss << "XAie_Lock(" << id << "," << val << ")";
  rss << "XAie_LockInit(" << id << "," << val << ")";
  return str;
}
static std::string tileLockStr(int id, int val) {
  return tileLockStr(std::to_string(id), std::to_string(val));
}
static std::string packetStr(StringRef id, StringRef type) {
  std::string str;
  llvm::raw_string_ostream rss(str);
  rss << "XAie_PacketInit(" << id << "," << type << ")";
  return str;
}
static std::string packetStr(int id, int type) {
  return packetStr(std::to_string(id), std::to_string(type));
}

mlir::LogicalResult AIETranslateToXAIEV2(ModuleOp module, raw_ostream &output) {
  StringRef enable = "XAIE_ENABLE";
  StringRef disable = "XAIE_DISABLE";
  StringRef resetDisable = "XAIE_RESETDISABLE";
  StringRef deviceInst    = "DevInst"; // TODO
  StringRef deviceInstRef = "&(DevInst)"; // TODO

  DenseMap<std::pair<int, int>, Operation *> tiles;
  DenseMap<Operation *, CoreOp> cores;
  DenseMap<Operation *, MemOp> mems;
  DenseMap<std::pair<Operation *, int>, LockOp> locks;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
  DenseMap<Operation *, SwitchboxOp> switchboxes;

  NetlistAnalysis NL(module, tiles, cores, mems, locks, buffers, switchboxes);
  NL.collectTiles(tiles);
  NL.collectBuffers(buffers);

  //---------------------------------------------------------------------------
  // mlir_configure_cores
  //---------------------------------------------------------------------------
  output << "void mlir_configure_cores() {\n";
  // Reset each core.  Load the corresponding ELF file, if necessary.
  for (auto tileOp : module.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    if (tileOp.isShimTile()) {
      // Resets no needed with V2 kernel driver
    } else {
      // Resets no needed with V2 kernel driver
      if (auto coreOp = tileOp.getCoreOp()) {
        std::string fileName;
        if (auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file")) {
          fileName = std::string(fileAttr.getValue());
        } else {
          fileName = std::string("core_") + std::to_string(col) + "_" +
                     std::to_string(row) + ".elf";
        }
        output << "{\n"
               << "AieRC RC = XAie_LoadElf("
               << deviceInstRef << ", "
               << tileLocStr(col, row)
               << ", "
               << "(const char*)\"" << fileName << "\",0);\n";
        output << "if (RC != XAIE_OK)\n"
               << "    printf(\"Failed to load elf for Core[%d,%d], ret is %d\\n\", "
               << std::to_string(col) << ", " << std::to_string(row) << ", RC);\n"
               << "assert(RC == XAIE_OK);\n"
               << "}\n";
      }
    }
  }
  output << "} // mlir_configure_cores\n\n";

  //---------------------------------------------------------------------------
  // mlir_start_cores
  //---------------------------------------------------------------------------
  output << "void mlir_start_cores() {\n";
  // Start execution of all the cores.
  for (auto tileOp : module.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    if (!tileOp.isShimTile()) {
      output << "XAie_CoreUnreset("
             << deviceInstRef << ", "
             << tileLocStr(col, row)
             << ");\n";
      output << "XAie_CoreEnable("
             << deviceInstRef << ", "
             << tileLocStr(col, row)
             << ");\n";
    }
  }
  output << "} // mlir_start_cores\n\n";

  //---------------------------------------------------------------------------
  // mlir_configure_dmas
  //---------------------------------------------------------------------------
  output << "void mlir_configure_dmas() {\n";

  // DMA configuration
  // AieRC XAie_DmaDescInit(XAie_DevInst *DevInst, XAie_DmaDesc *DmaDesc, XAie_LocType Loc);
  // AieRC XAie_DmaSetLock(XAie_DmaDesc *DmaDesc, XAie_Lock Acq, XAie_Lock Rel);
  // AieRC XAie_DmaSetPkt(XAie_DmaDesc *DmaDesc, XAie_Packet Pkt);
  // AieRC XAie_DmaSetOutofOrderBdId(XAie_DmaDesc *DmaDesc, u8 OutofOrderBdId);
  // AieRC XAie_DmaSetDoubleBuffer(XAie_DmaDesc *DmaDesc, u64 Addr, XAie_Lock Acq, XAie_Lock Rel);
  // AieRC XAie_DmaSetAddrLen(XAie_DmaDesc *DmaDesc, u64 Addr, u32 Len);
  // AieRC XAie_DmaSetMultiDimAddr(XAie_DmaDesc *DmaDesc, XAie_DmaTensor *Tensor, u64 Addr, u32 Len);
  // AieRC XAie_DmaEnableCompression(XAie_DmaDesc *DmaDesc);
  // AieRC XAie_DmaSetNextBd(XAie_DmaDesc *DmaDesc, u8 NextBd, u8 EnableNextBd);
  // AieRC XAie_DmaEnableBd(XAie_DmaDesc *DmaDesc);
  // AieRC XAie_DmaDisableBd(XAie_DmaDesc *DmaDesc);
  // AieRC XAie_DmaSetAxi(XAie_DmaDesc *DmaDesc, u8 Smid, u8 BurstLen, u8 Qos,u8 Cache, u8 Secure);
  // AieRC XAie_DmaSetInterleaveEnable(XAie_DmaDesc *DmaDesc, u8 DoubleBuff, u8 IntrleaveCount, u16 IntrleaveCurr);
  // AieRC XAie_DmaWriteBd(XAie_DevInst *DevInst, XAie_DmaDesc *DmaDesc, XAie_LocType Loc, u8 BdNum);
  
  // AieRC XAie_DmaChannelResetAll(XAie_DevInst *DevInst, XAie_LocType Loc, XAie_DmaChReset Reset);
  // AieRC XAie_DmaChannelReset(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, XAie_DmaChReset Reset);
  // AieRC XAie_DmaChannelPauseStream(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 Pause);
  // AieRC XAie_DmaChannelPauseMem(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum XAie_DmaDirection Dir, u8 Pause);
  // AieRC XAie_DmaChannelConfig(XAie_DevInst *DevInst, XAie_DmaDesc *DmaDesc, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 RepeatCount, u8 EnTokenIssue, u8 ControllerId);
  // AieRC XAie_DmaChannelPushBdToQueue(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 BdNum);
  // AieRC XAie_DmaChannelEnable(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir);
  // AieRC XAie_DmaChannelDisable(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir);
  for (auto memOp : module.getOps<MemOp>()) {
    int col = memOp.colIndex();
    int row = memOp.rowIndex();
    // Reset not needed with V2 kernel driver

    DenseMap<Block *, int> blockMap;

    {
      // Assign each block a BD number
      int bdNum = 0;
      for (auto &block : memOp.body()) {
        if (!block.getOps<DMABDOp>().empty()) {
          blockMap[&block] = bdNum;
          bdNum++;
        }
      }
    }
    for (auto &block : memOp.body()) {
      bool foundBdPacket = false;
      int packetType = 0;
      int packetID = 0;
      bool foundBd = false;
      int lenA = 0;
      int lenB = 0;
      int bytesA = 0;
      int bytesB = 0;
      int offsetA = 0;
      int offsetB = 0;
      int BaseAddrA = 0;
      int BaseAddrB = 0;
      bool hasA = false;
      bool hasB = false;
      StringRef bufA = "0";
      StringRef bufB = "0";
      StringRef AbMode = disable;
      StringRef FifoMode = disable; // FIXME: when to enable FIFO mode?
      for (auto op : block.getOps<DMABDOp>()) {
        foundBd = true;
        ShapedType bufferType =
            op.buffer().getType().cast<::mlir::MemRefType>();
        if (op.isA()) {
          BaseAddrA = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
          lenA = op.getLenValue();
          bytesA = bufferType.getElementTypeBitWidth() / 8;
          offsetA = op.getOffsetValue();
          bufA = "XAIEDMA_TILE_BD_ADDRA";
          hasA = true;
        }
        if (op.isB()) {
          BaseAddrB = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
          lenB = op.getLenValue();
          bytesB = bufferType.getElementTypeBitWidth() / 8;
          offsetB = op.getOffsetValue();
          bufB = "XAIEDMA_TILE_BD_ADDRB";
          hasB = true;
        }
      }

      if (hasA && hasB) {
        AbMode = enable;
        if (lenA != lenB)
          llvm::errs() << "ABmode must have matching lengths.\n";
        if (bytesA != bytesB)
          llvm::errs() << "ABmode must have matching element data types.\n";
      }
      int acqValue = 0, relValue = 0;
      StringRef acqEnable = disable;
      StringRef relEnable = disable;
      int lockID;
      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
        lockID = lock.getLockID();
        if (op.acquire()) {
          acqEnable = enable;
          acqValue = op.getLockValue();
        } else if (op.release()) {
          relEnable = enable;
          relValue = op.getLockValue();
        }
      }

      for (auto op : block.getOps<DMABDPACKETOp>()) {
        foundBdPacket = true;
        packetType = op.getPacketType();
        packetID = op.getPacketID();
      }

      int bdNum = blockMap[&block];
      if (foundBd) {
        // TODO AB mode separated

        // TODO For now, we are going to name each dma desc with loc and bd 
        // which we assume is unique. This is strictly not enforced but in 
        // practice, this is true
        output << "XAie_DmaDesc "
               << tileDMAInstStr(col, row, bdNum) << ";\n";
        output << "XAie_DmaDescInit("
               << deviceInstRef << ", "
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << tileLocStr(col, row) << ");\n";
        output << "XAie_DmaSetLock("
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << "XAie_LockInit(" << lockID << "," << acqValue << "),"
               << "XAie_LockInit(" << lockID << "," << relValue << "));\n";
        output << "XAie_DmaSetAddrLen("
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << " /* addrA */ " << "0x" 
               << llvm::utohexstr(BaseAddrA + offsetA) << ", "
               << " /* len */ "  << lenA << " * " << bytesA << ");\n";
                  
        if(block.getNumSuccessors() > 0) {
          Block *nextBlock = block.getSuccessors()[0]; // should have only one 
                                                       // successor block
          int nextBdNum = blockMap[nextBlock];
          output << "XAie_DmaSetNextBd(" 
                 << tileDMAInstRefStr(col, row, bdNum) << ", "
                 << " /* bd */ " << bdNum << ", "
                 << " /* nextbd */ " << nextBdNum << ");\n";
        }
        if (foundBdPacket) {
          output << "XAie_DmaSetPkt(" 
                 << tileDMAInstRefStr(col, row, bdNum) << ", "
                 << packetStr(packetID, packetType) << ");\n";
        }
        output << "XAie_DmaEnableBd(" 
               << tileDMAInstRefStr(col, row, bdNum) << ");\n";
        output << "XAie_DmaWriteBd("  << deviceInstRef << ", "
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << tileLocStr(col, row) << ", "
               << " /* bd */ " << bdNum << ");\n";
      }
    }

    for (auto &block : memOp.body()) {
      for (auto op : block.getOps<DMAStartOp>()) {
        int bdNum = blockMap[op.dest()];

        llvm::StringRef dmaChan = stringifyDMAChan(op.dmaChan());
        llvm::StringRef dmaDir  = dmaChan.substr(0,4);
        llvm::StringRef chNum   = dmaChan.substr(4,1);

        output << "XAie_DmaChannelPushBdToQueue("
               << deviceInstRef << ", "
               << tileLocStr(col, row) << ", "
               << "/* ChNum */" << chNum << ", "
               // TODO hack until physical dialect changes
               << "/* dmaDir */ DMA_" << dmaDir << ", " 
               << "/* BdNum */" << bdNum << ");\n";
        output << "XAie_DmaChannelEnable("
               << deviceInstRef << ", "
               << tileLocStr(col, row) << ", "
               << "/* ChNum */ " << chNum << ", "
               // TODO hack until physical dialect changes
               << "/* dmaDir */ DMA_" << dmaDir << ");\n"; 
      }
    }
  }
  // ShimDMA Config
  int index = 0;
  for (auto op : module.getOps<ShimDMAOp>()) {
    int col = op.colIndex();
    int row = op.rowIndex();

    DenseMap<Block *, int> blockMap;

    {
      // Assign each block a BD number
      int bdNum = 0;
      for (auto &block : op.body()) {
        if (!block.getOps<DMABDOp>().empty()) {
          blockMap[&block] = bdNum;
          bdNum++;
        }
      }
    }
    for (auto &block : op.body()) {
      bool foundBd = false;
      int len = 0;
      uint64_t bytes = 0;
      uint64_t offset = 0;
      uint64_t BaseAddr = 0;

      for (auto op : block.getOps<DMABDOp>()) {
        foundBd = true;
        len = op.getLenValue();
        ShapedType bufferType =
            op.buffer().getType().cast<::mlir::MemRefType>();
        bytes = bufferType.getElementTypeBitWidth() / 8;
        BaseAddr = NL.getBufferBaseAddress(op.buffer().getDefiningOp());
        offset = op.getOffsetValue();
      }

      int acqValue = 0, relValue = 0;
      bool hasLock = false;
      StringRef acqEnable = disable;
      StringRef relEnable = disable;
      int lockID = 0;
      for (auto op : block.getOps<UseLockOp>()) {
        LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
        lockID = lock.getLockID();
        hasLock = true;
        if (op.acquire()) {
          acqEnable = enable;
          acqValue = op.getLockValue();
        } else if (op.release()) {
          relEnable = enable;
          relValue = op.getLockValue();
        }
      }

      int bdNum = blockMap[&block];
      if (foundBd) {
        output << "XAie_DmaDescInit(" << deviceInstRef << ", "
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << tileLocStr(col, row) << ");\n";

        if (hasLock)
          output << "XAie_DmaSetLock("
                 << tileDMAInstRefStr(col, row, bdNum) << ", "
                 << "XAie_LockInit(" << lockID << "," << acqValue << "),"
                 << "XAie_LockInit(" << lockID << "," << relValue << "));\n";
        uint64_t address = BaseAddr + offset;
        output << "XAie_DmaSetAddrLen("
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << " /* addr */ " << "0x" << llvm::utohexstr(address) << ", "
               << " /* len */ "  << len << " * " << bytes << ");\n";

        output << "XAie_DmaSetAxi("
               << deviceInstRef << ", "
               << "/* smid */ 0, "
               << "/* burstlen */ 4, "
               << "/* QoS */ 0 , "
               << "/* Cache */ 0, "
               << "/* Secure */ " << enable << ");\n";

        if (block.getNumSuccessors() > 0) {
          Block *nextBlock = block.getSuccessors()[0]; // should have only one
                                                       // successor block
          int nextBdNum = blockMap[nextBlock];
          // void XAieDma_ShimBdSetNext(XAieDma_Shim *DmaInstPtr, u8
          // BdNum, u8 NextBd);
          output << "XAie_DmaSetNextBd("
                 << tileDMAInstRefStr(col, row, bdNum) << ", "
                 << " /* bd */ " << bdNum << ", "
                 << " /* nextbd */ " << nextBdNum << ");\n";
        }
        output << "XAie_DmaEnableBd("
               << tileDMAInstRefStr(col, row, bdNum) << ");\n";
        output << "XAie_DmaWriteBd(" << deviceInstRef << ", "
               << tileDMAInstRefStr(col, row, bdNum) << ", "
               << tileLocStr(col, row) << ", "
               << " /* bd */ " << bdNum << ");\n";
      }
    }

    for (auto &block : op.body()) {
      for (auto op : block.getOps<DMAStartOp>()) {
        int bdNum = blockMap[op.dest()];

        llvm::StringRef dmaChan = stringifyDMAChan(op.dmaChan());
        llvm::StringRef dmaDir  = dmaChan.substr(0,4);
        llvm::StringRef chNum   = dmaChan.substr(4,1);

        output << "XAie_DmaChannelPushBdToQueue(" << deviceInstRef << ", "
               << tileLocStr(col, row) << ", "
               << "/* ChNum */" << chNum << ", "
               // TODO hack until physical dialect changes
               << "/* dmaDir */ DMA_" << dmaDir << ", " 
               << "/* BdNum */" << bdNum << ");\n";
        output << "XAie_DmaChannelEnable(" << deviceInstRef << ", "
               << tileLocStr(col, row) << ", "
               << "/* ChNum */ " << chNum << ", "
               // TODO hack until physical dialect changes
               << "/* dmaDir */ DMA_" << dmaDir << ");\n"; 
      }
    }
  }
  output << "} // mlir_configure_dmas\n\n";

  //---------------------------------------------------------------------------
  // mlir_initialize_locks
  //---------------------------------------------------------------------------
  output << "void mlir_initialize_locks() {\n";
  // Lock configuration
  for (auto op : module.getOps<UseLockOp>()) {
    int lockVal = op.getLockValue();
    int timeOut = op.getTimeout();
    LockOp lock = dyn_cast<LockOp>(op.lock().getDefiningOp());
    TileOp tile = dyn_cast<TileOp>(lock.tile().getDefiningOp());
    int col = tile.colIndex();
    int row = tile.rowIndex();
    int lockID = lock.getLockID();
    if (op.acquire()) {
      output << "XAie_LockAcquire(" << deviceInstRef << ", "
             << tileLocStr(col, row) << ", "
             << tileLockStr(lockID,lockVal) << ", "
             << timeOut << ");\n";
    } else if (op.release()) {
      output << "XAie_LockRelease(" << deviceInstRef << ", "
             << tileLocStr(col, row) << ", "
             << tileLockStr(lockID, lockVal) << ", " 
             << timeOut << ");\n";
    }
  }
  output << "} // mlir_initialize_locks\n";

  //---------------------------------------------------------------------------
  // mlir_configure_switchboxes
  //---------------------------------------------------------------------------
  output << "void mlir_configure_switchboxes() {\n";
  output << "  int x, y;\n";

  // StreamSwitch (switchbox) configuration
  for (auto switchboxOp : module.getOps<SwitchboxOp>()) {
    Region &r = switchboxOp.connections();
    Block &b = r.front();
    bool isEmpty = b.getOps<ConnectOp>().empty() &&
                   b.getOps<MasterSetOp>().empty() &&
                   b.getOps<PacketRulesOp>().empty();
    bool isParam = false;

    if (isa<TileOp>(switchboxOp.tile().getDefiningOp())) {
      int col = switchboxOp.colIndex();
      int row = switchboxOp.rowIndex();
      if (!isEmpty) {
        output << "// Core Stream Switch column " << col << " row " << row
               << "\n";
        output << "x = " << col << ";\n";
        output << "y = " << row << ";\n";
      }
    } else if (AIE::SelectOp sel = dyn_cast<AIE::SelectOp>(
                   switchboxOp.tile().getDefiningOp())) {
      // parameterize streamswitch's configuration
      isParam = true;
      HerdOp sourceHerd = dyn_cast<HerdOp>(sel.startHerd().getDefiningOp());
      std::string sourceHerdName(sourceHerd.name().getValue());

      IterOp iterX = dyn_cast<IterOp>(sel.iterX().getDefiningOp());
      IterOp iterY = dyn_cast<IterOp>(sel.iterY().getDefiningOp());
      int startXValue = iterX.getStartValue();
      int endXValue = iterX.getEndValue();
      int strideXValue = iterX.getStrideValue();
      int startYValue = iterY.getStartValue();
      int endYValue = iterY.getEndValue();
      int strideYValue = iterY.getStrideValue();

      std::string startX(sourceHerdName + "_X + " +
                         std::to_string(startXValue));
      std::string endX(sourceHerdName + "_X + " + std::to_string(endXValue));
      std::string startY(sourceHerdName + "_Y + " +
                         std::to_string(startYValue));
      std::string endY(sourceHerdName + "_Y + " + std::to_string(endYValue));

      output << "for (x = " << startX << "; x < " << endX
             << "; x += " << strideXValue << ") {\n";
      output << "for (y = " << startY << "; y < " << endY
             << "; y += " << strideYValue << ") {\n";
    }

    for (auto connectOp : b.getOps<ConnectOp>()) {
      output << "XAie_StrmConnCctEnable(" << deviceInstRef << ", "
             << tileLocStr("x", "y") << ", "
             << stringifyWireBundle(connectOp.sourceBundle()).upper() << ", "
             << connectOp.sourceIndex() << ", "
             << stringifyWireBundle(connectOp.destBundle()).upper() << ", "
             << connectOp.destIndex() << ");\n";
    }

    for (auto connectOp : b.getOps<MasterSetOp>()) {
      int mask = 0;
      int arbiter = -1;
      for (auto val : connectOp.amsels()) {
        AMSelOp amsel = dyn_cast<AMSelOp>(val.getDefiningOp());
        arbiter = amsel.arbiterIndex();
        int msel = amsel.getMselValue();
        mask |= (1 << msel);
      }

      output << "XAie_StrmPktSwMstrPortEnable(" << deviceInstRef << ", "
             << tileLocStr("x", "y") << ", "
             << stringifyWireBundle(connectOp.destBundle()).upper() << ", "
             << connectOp.destIndex() << ", "
             << "/* drop_header */ " << "XAIE_SS_PKT_DROP_HEADER" << ", " // TODO is this right default???
             << "/* arbiter */ " << arbiter << ", "
             << "/* MSelEn */ 1);\n"; // TODO do I need mask instead???
    }

    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      int slot = 0;
      Block &block = connectOp.rules().front();
      for (auto slotOp : block.getOps<PacketRuleOp>()) {
        AMSelOp amselOp = dyn_cast<AMSelOp>(slotOp.amsel().getDefiningOp());
        int arbiter = amselOp.arbiterIndex();
        int msel = amselOp.getMselValue();
        output << "XAie_StrmPktSwSlavePortEnable(" << deviceInstRef << ", "
               << tileLocStr("x", "y") << ", "
               << stringifyWireBundle(connectOp.sourceBundle()).upper() << ", "
               << connectOp.sourceIndex() << ");\n";

        // TODO Need to better define packet id,type used here
        output << "XAie_StrmPktSwSlaveSlotEnable(" << deviceInstRef << ", "
               << tileLocStr("x", "y") << ", "
               << stringifyWireBundle(connectOp.sourceBundle()).upper() << ", "
               << connectOp.sourceIndex() << ", "
               << "/* slot */ " << slot << ", "
               << "/* packet */ " << packetStr(slotOp.valueInt(), /*type*/ 0) << ", "
               << "/* mask */ " << "0x" << llvm::utohexstr(slotOp.maskInt()) << ", "
               << "/* msel */ " << msel << ", "
               << "/* arbiter */ " << arbiter << ");\n";
        slot++;
      }
    }

    if (isParam) {
      output << "}\n";
      output << "}\n";
    }
  }
  for (auto op : module.getOps<ShimMuxOp>()) {
    Region &r = op.connections();
    Block &b = r.front();
    bool isEmpty = b.getOps<ConnectOp>().empty();

    if (isa<TileOp>(op.tile().getDefiningOp())) {
      int col = op.colIndex();
      int row = op.rowIndex();
      if (!isEmpty) {
        output << "// ShimMux column " << col << " row " << row << "\n";
        output << "// NOTE ShimMux always connects from the south as "
               << "directions are defined relative to the tile stream "
               << "switch\n";
        output << "x = " << col << ";\n";
        output << "y = " << row << ";\n";
      }
    }

    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourceBundle() == WireBundle::North) {
        // demux!
        output << "XAie_EnableShimDmaToAieStrmPort(" << deviceInstRef << ", "
               << tileLocStr("x", "y") << ", "
               << stringifyWireBundle(connectOp.sourceBundle()).back() << ");\n";
      } else if (connectOp.destBundle() == WireBundle::North) {
        // mux
        output << "XAie_EnableShimAieToDmaStrmPort(" << deviceInstRef << ", "
               << tileLocStr("x", "y") << ", "
               << stringifyWireBundle(connectOp.sourceBundle()).back() << ");\n";
      }
    }
  }
  for (auto switchboxOp : module.getOps<ShimSwitchboxOp>()) {
    Region &r = switchboxOp.connections();
    Block &b = r.front();
    bool isEmpty = b.getOps<ConnectOp>().empty();
    int col = switchboxOp.col();
    if (!isEmpty) {
      output << "// Shim Switch column " << col << "\n";
    }
    for (auto connectOp : b.getOps<ConnectOp>()) {
      output << "XAie_StrmConnCctEnable(" << deviceInstRef << ", "
             << tileLocStr(col, 0) << ", "
             << stringifyWireBundle(connectOp.sourceBundle()).upper() << ", "
             << connectOp.sourceIndex() << ", "
             << stringifyWireBundle(connectOp.destBundle()).upper() << ", "
             << connectOp.destIndex() << ");\n";
    }
  }

  output << "} // mlir_configure_switchboxes\n\n";

  //---------------------------------------------------------------------------
  // Output Buffer Accessors
  //---------------------------------------------------------------------------
  for (auto tile : tiles) {
    Operation *tileOp = tile.second;
    std::pair<int, int> coord = NL.getCoord(tileOp);
    int col = coord.first;
    int row = coord.second;
    auto loc = tileLocStr(col,row);

    auto bufferAccessor = [&](Optional<TileID> tile, BufferOp buf) {
      // int32_t mlir_read_buffer_a13(int index) {
      // void mlir_write_buffer_a13(int index, int32_t value) {
      std::string bufName(buf.name().getValue());
      Type t = buf.getType();
      Type et;
      std::string typestr;
      if (auto memrefType = t.dyn_cast<MemRefType>()) {
        et = memrefType.getElementType();
        if (et.isInteger(32))
          typestr = "int32_t";
        else if (et.isF32())
          typestr = "float";
        else {
          output << "// buffer " << bufName << " with unsupported type " << t
                 << ";\n";
          return; // Unsupported type
        }

      } else {
        output << "// buffer " << bufName << " with unsupported type " << t
               << ";\n";
        return; // Unsupported type
      }

      output << "const int " << bufName
             << "_offset = " << NL.getBufferBaseAddress(buf) << ";\n";
      output << typestr << " mlir_read_buffer_" << bufName << "(int index) {\n";
      output << "u32 value; auto rc = XAie_DataMemRdWord(" << deviceInstRef << ", " << loc
             << ", " << bufName << "_offset + (index*4), &value);\n";
      if (et.isInteger(32))
        output << "  return value;\n";
      else if (et.isF32()) {
        output << "  union caster { int32_t i; float f; };\n";
        output << "  caster c; c.i = value;\n";
        output << "  return c.f;\n";
      }
      output << "}\n";
      output << "void mlir_write_buffer_" << bufName << "(int index, "
             << typestr << " value) {\n";
      if (et.isInteger(32))
        output << "  int32_t int_value = value;\n";
      else if (et.isF32()) {
        output << "  union caster { int32_t i; float f; };\n";
        output << "  caster c; c.f = value;\n";
        output << "  int32_t int_value = c.i;\n";
      }
      output << "u32 rc =    XAie_DataMemWrWord(" << deviceInstRef << ", " << loc << ", "
             << bufName << "_offset + (index*4), int_value);\n";
      output << "}\n";
    };

    // if(tiles.count(tile.getValue()))
    for (auto buf : buffers[tileOp])
      bufferAccessor(coord, buf);
    // };
  }
  return success();
}
} // namespace AIE
} // namespace xilinx
