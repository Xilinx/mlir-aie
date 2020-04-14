// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aie;

static TranslateFromMLIRRegistration
    registration("aie-generate-xaie", [](ModuleOp module, raw_ostream &output) {
        // XAieTile_StrmConnectCct(&(TileInst[col+i][row]),
        //                         XAIETILE_STRSW_SPORT_TRACE((&(TileInst[col+i][row])), 1),
        //                         XAIETILE_STRSW_MPORT_NORTH((&(TileInst[col+i][row])), 0), XAIE_ENABLE);
        for(auto switchboxOp : module.getOps<SwitchboxOp>()) {
          Region &r = switchboxOp.connections();
          Block &b = r.front();
          bool isEmpty = b.getOps<ConnectOp>().empty();
          int col = switchboxOp.col().getZExtValue();
          int row = switchboxOp.row().getZExtValue();
          if(!isEmpty) {
            output << "// Core Stream Switch column " << col << " row " << row << "\n";
            output << "{XAieGbl_Tile *inst = &(TileInst["
                     << col << "][" << row + 1 << "]);\n";
          }
          for (auto connectOp : b.getOps<ConnectOp>()) {
            output << "  XAieTile_StrmConnectCct(inst,\n";
            output << "\tXAIETILE_STRSW_SPORT_" << stringifyWireBundle(connectOp.sourceBundle()).upper() << "(inst, " << connectOp.sourceIndex() << "),\n";
            output << "\tXAIETILE_STRSW_MPORT_" << stringifyWireBundle(connectOp.destBundle()).upper() << "(inst, " << connectOp.destIndex() << "),\n";
            output << "\tXAIE_ENABLE);\n";
          }
          std::vector<int> mselForArbiter;
          DenseMap<Operation *, int> mselForMasterSet;
          for (auto connectOp : b.getOps<MasterSetOp>()) {
            int arbiter = connectOp.arbiterIndex();
            if(arbiter >= mselForArbiter.size())
              mselForArbiter.resize(arbiter + 1);
            int msel = mselForArbiter[arbiter]++;
            mselForMasterSet[connectOp.getOperation()] = msel;
            output << "  XAieTile_StrmConfigMstr(inst,\n";
            output << "\tXAIETILE_STRSW_MPORT_" << stringifyWireBundle(connectOp.destBundle()).upper() << "(inst, " << connectOp.destIndex() << "),\n";
            output << "\tXAIE_ENABLE,\n";
            output << "\tXAIE_ENABLE,\n";
            output << "\tXAIETILE_STRSW_MPORT_CFGPKT(inst,\n";
            output << "\t\tXAIETILE_STRSW_MPORT_" << stringifyWireBundle(connectOp.destBundle()).upper() << "(inst, " << connectOp.destIndex() << "),\n";
            output << "\t\tXAIE_DISABLE /*drop_header*/,\n";
            output << "\t\t" << msel << "/*msel*/,\n";
            output << "\t\t" << connectOp.arbiter() << "/*arbiter*/);\n";
          }
          for (auto connectOp : b.getOps<PacketRulesOp>()) {
            int slot = 0;
            output << "PacketRules:\n";
            Block &block = connectOp.rules().front();
            for (auto slotOp : block.getOps<PacketRuleOp>()) {
              Operation *op = slotOp.masterset().getDefiningOp();
              MasterSetOp masterSetOp = dyn_cast<MasterSetOp>(op);
              output << "  XAieTile_StrmConfigSlvSlot(inst,\n";
              output << "\tXAIETILE_STRSW_SPORT_" << stringifyWireBundle(connectOp.sourceBundle()).upper() << "(inst, " << connectOp.sourceIndex() << "),\n";
              output << "\t" << slot << "/*slot*/,\n";
              output << "\tXAIE_ENABLE,\n";
              output << "\tXAIETILE_STRSW_SLVSLOT_CFG(inst,\n";
              output << "\t\tXAIETILE_STRSW_SPORT_" << stringifyWireBundle(connectOp.sourceBundle()).upper() << "(inst, " << connectOp.sourceIndex() << "),\n";
              output << "\t\t" << slot << "/*slot*/,\n";
              output << "\t\t" << slotOp.valueInt() << "/*ID value*/,\n";
              output << "\t\t" << slotOp.maskInt() << "/*mask*/,\n";
              output << "\t\tXAIE_ENABLE,\n";
              output << "\t\t" << mselForMasterSet[op] << "/*msel*/,\n";
              output << "\t\t" << masterSetOp.arbiter() << "/*arbiter*/);\n";
              slot++;
            }
          }

          if(!isEmpty) {
            output << "}\n";
          }
        }
        for(auto switchboxOp : module.getOps<ShimSwitchboxOp>()) {
          Region &r = switchboxOp.connections();
          Block &b = r.front();
          bool isEmpty = b.getOps<ConnectOp>().empty();
          int col = switchboxOp.col().getZExtValue();
          if(!isEmpty) {
            output << "// Shim Switch column " << col << "\n";
            output << "{XAieGbl_Tile *inst = &(TileInst["
                   << col << "]["
                   << "0" << "]);\n";
          }
          for (auto connectOp : b.getOps<ConnectOp>()) {
            output << "  XAieTile_StrmConnectCct(inst,\n";
            output << "\tXAIETILE_STRSW_SPORT_" << stringifyWireBundle(connectOp.sourceBundle()).upper() << "(inst, " << connectOp.sourceIndex() << "),\n";
            output << "\tXAIETILE_STRSW_MPORT_" << stringifyWireBundle(connectOp.destBundle()).upper() << "(inst, " << connectOp.destIndex() << "),\n";
            output << "\tXAIE_ENABLE);\n";
          }
          if(!isEmpty) {
            output << "}\n";
          }
        }
        return success();
      });
