//===- AIEFindFlows.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

typedef std::pair<int, int> MaskValue;
typedef std::pair<Operation *, Port> PortConnection;
typedef std::pair<Port, MaskValue> PortMaskValue;
typedef std::pair<PortConnection, MaskValue> PacketConnection;
typedef std::pair<int, int> Coord;
typedef std::pair<Port, int> PortID;
typedef std::pair<Coord, PortID> PacketFlowEndpoint;
typedef std::pair<PacketFlowEndpoint, PacketFlowEndpoint> PacketFlow;

class ConnectivityAnalysis {
  ModuleOp &module;

public:
  // At each switchbox coordinate, track which flows are in transit
  DenseMap<Coord, SmallVector<int, 8>* > in_transit_pkt_flow_ids;

  // When delivered to a path endpoint, e.g. Core or DMA, add to this list
  DenseMap<Coord, SmallVector<PortID, 8>* > delivered_pkt_flow_ids;
  int MAX_COL = 0, MAX_ROW = 0;

  ConnectivityAnalysis(ModuleOp &m) : module(m) {
    for (auto tile : m.getOps<TileOp>()) {
      if (tile.colIndex() > MAX_COL)
        MAX_COL = tile.colIndex();
      if (tile.rowIndex() > MAX_ROW)
        MAX_ROW = tile.rowIndex();
    }
    resetInTransitIDs();
    resetDeliveredIDs();
    LLVM_DEBUG(llvm::dbgs() << "MAX_COL = " << MAX_COL << "\n");
    LLVM_DEBUG(llvm::dbgs() << "MAX_ROW = " << MAX_ROW << "\n");
  }

  void resetInTransitIDs() {
    in_transit_pkt_flow_ids.clear();
    for (int col = 0; col <= MAX_COL; col++)
      for (int row = 0; row <= MAX_ROW; row++)
        in_transit_pkt_flow_ids[std::make_pair(col, row)] = new SmallVector<int, 8>();
  }

  void resetDeliveredIDs() {
    delivered_pkt_flow_ids.clear();
    for (int col = 0; col <= MAX_COL; col++)
      for (int row = 0; row <= MAX_ROW; row++)
        delivered_pkt_flow_ids[std::make_pair(col, row)] = new SmallVector<PortID, 8>();
  }

  // Debugging print functions
  void print_in_transit_pkt_flow_ids() {
    LLVM_DEBUG(llvm::dbgs() << " Packet IDs in transit:\n");
    for (auto  item : in_transit_pkt_flow_ids) {
      if (item.second->size() > 0) {
        LLVM_DEBUG(llvm::dbgs() << "(" << item.first.first << ", " << item.first.second << "): ");
        for( int ID : *(item.second) )
          LLVM_DEBUG(llvm::dbgs() << ID << ", ");
        LLVM_DEBUG(llvm::dbgs() << "\n");
      }
    }
  }

  static void printPortID(PortID p) {
          LLVM_DEBUG(llvm::dbgs() << "PortID: ID(" << p.second << ") delivered to: "
                    << stringifyWireBundle(p.first.first)
                    << ":" << p.first.second << "\n");
  }

  void print_delivered_pkt_flow_ids() {
    LLVM_DEBUG(llvm::dbgs() << " Packet IDs delivered:\n");
    for (auto  item : delivered_pkt_flow_ids) {
      if (item.second->size() > 0) {
        LLVM_DEBUG(llvm::dbgs() << "To destination tile: (" << item.first.first << ", " << item.first.second << "):\n");
        for(PortID p : *(item.second) )
          printPortID(p);
        LLVM_DEBUG(llvm::dbgs() << "\n");
      }
    }
  }

  static void printPortConnection(PortConnection pc) {
    LLVM_DEBUG(pc.first->dump());
    Port port = pc.second;
    LLVM_DEBUG(llvm::dbgs() << stringifyWireBundle(port.first) << " " << (int)port.second << " ");
  }

  static void printMaskValue(MaskValue v) {
    LLVM_DEBUG(llvm::dbgs() << "mask: (" << v.first << ", " << v.second << ") ");
  }

  static void printPacketConnection(PacketConnection const &p) {
    LLVM_DEBUG(llvm::dbgs() << "PacketConnection: ");
    printPortConnection(p.first);
    printMaskValue(p.second);
  }

  static std::pair<int, int> getNextCoord(std::pair<int, int> coord, WireBundle move) {
  int x = coord.first;
  int y = coord.second;
  if (move == WireBundle::East) {
    return std::make_pair(x+1, y);
  } else if (move == WireBundle::West) {
    return std::make_pair(x-1, y);
  } else if (move == WireBundle::North) {
    return std::make_pair(x, y+1);
  } else if (move == WireBundle::South) {
    return std::make_pair(x, y-1);
  } else return std::make_pair(x, y);
}

private:
  llvm::Optional<PortConnection>
  getConnectionThroughWire(Operation *op, Port masterPort) const {
    for (auto wireOp : module.getOps<WireOp>()) {
      if (wireOp.getSource().getDefiningOp() == op &&
          wireOp.getSourceBundle() == masterPort.first) {
        Operation *other = wireOp.getDest().getDefiningOp();
        Port otherPort =
            std::make_pair(wireOp.getDestBundle(), masterPort.second);
        return std::make_pair(other, otherPort);
      }
      if (wireOp.getDest().getDefiningOp() == op &&
          wireOp.getDestBundle() == masterPort.first) {
        Operation *other = wireOp.getSource().getDefiningOp();
        Port otherPort =
            std::make_pair(wireOp.getSourceBundle(), masterPort.second);
        return std::make_pair(other, otherPort);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "*** Missing Wire!!!\n");
    return std::nullopt;
  }

  std::vector<PortMaskValue>
  getConnectionsThroughSwitchbox(Operation* op, Region &r, Port sourcePort) const {
    std::pair<int, int> sb_coord;
    if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(op))
      sb_coord = std::make_pair(switchOp.colIndex(), switchOp.rowIndex());
    else if(auto switchOp = dyn_cast_or_null<ShimMuxOp>(op))
      sb_coord = std::make_pair(switchOp.colIndex(), switchOp.rowIndex());
    else { // TODO: better way to emit an error? 
      LLVM_DEBUG(llvm::dbgs() << "ERROR: getConnectionsThroughSwitchbox():"
                  << "op is not a valid SwitchboxOp or ShimMuxOp!\n");
      std::exit(1);
    }

    Block &b = r.front();
    std::vector<PortMaskValue> portSet;
    
    // Circuit Switched flows
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        MaskValue maskValue = std::make_pair(0, 0);
        portSet.push_back(std::make_pair(connectOp.destPort(), maskValue));
      }
    }

    // Packet flows
    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Packet From: "
                   << stringifyWireBundle(connectOp.sourcePort().first) << " "
                   << (int)sourcePort.second << "\n");

        //auto ruleOps = connectOp.getRules().front().getOps<PacketRuleOp>();
        for (auto ruleOp : connectOp.getRules().front().getOps<PacketRuleOp>())
          for (auto masterSetOp : b.getOps<MasterSetOp>())
            for (Value amsel : masterSetOp.getAmsels())
            {
              if (ruleOp.getAmsel() == amsel) {
                MaskValue maskValue =
                    std::make_pair(ruleOp.maskInt(), ruleOp.valueInt());

                LLVM_DEBUG(llvm::dbgs() << "Packet IDs in transit at (" << sb_coord.first << ", " << sb_coord.second << "):\n");
                SmallVector<int, 8>* pkt_ids_at_sb = in_transit_pkt_flow_ids.at(sb_coord);
                for (int ID : *pkt_ids_at_sb)
                      LLVM_DEBUG(llvm::dbgs() << ID << ", ");
                LLVM_DEBUG(llvm::dbgs() << "\n");

                LLVM_DEBUG(llvm::dbgs()
                           << "Connects To: "
                           << stringifyWireBundle(masterSetOp.destPort().first)
                           << " " << masterSetOp.destPort().second 
                           << "\tMask: " << maskValue.first << "\tMatch Value: " << maskValue.second << "\n\n");

                int mask = maskValue.first;
                int match = maskValue.second;

                auto next_sb_coord = getNextCoord(sb_coord, masterSetOp.destPort().first);

                LLVM_DEBUG(llvm::dbgs() << "next_sb_coord: (" << next_sb_coord.first << ", "
                                          << next_sb_coord.second << ")\n");

                for(unsigned int ID = 0; ID < 32; ID++){
                  if (std::count(pkt_ids_at_sb->begin(), pkt_ids_at_sb->end(), ID)) {
                    if ((ID & mask) == unsigned(match & mask)) {
                      // Remove this ID from current switchbox list
                      for(unsigned int i = 0; i < pkt_ids_at_sb->size(); i++)
                        if((unsigned)(*pkt_ids_at_sb)[i] == ID)
                          pkt_ids_at_sb->erase(pkt_ids_at_sb->begin()+i);

                      // Add this ID to next switchbox list,
                      // or delivered list if reaching a DMA or Core
                      if(masterSetOp.destPort().first == WireBundle::DMA ||
                         masterSetOp.destPort().first == WireBundle::Core) {
                        LLVM_DEBUG(llvm::dbgs() << "\nFlow ID " << ID << " successfully delivered to: ("
                                  << next_sb_coord.first << ", " << next_sb_coord.second << ") " 
                                  << stringifyWireBundle(masterSetOp.destPort().first) << ":" 
                                  << masterSetOp.destPort().second << " : ");
                        PortID portID = std::make_pair(masterSetOp.destPort(), ID);
                        delivered_pkt_flow_ids.at(sb_coord)->push_back(portID);
                      } else {
                        in_transit_pkt_flow_ids.at(next_sb_coord)->push_back(ID);
                      }
                      LLVM_DEBUG(llvm::dbgs() << ID << ", ");
                    }
                  }
                }
                LLVM_DEBUG(llvm::dbgs() << "\n");
                portSet.push_back(
                    std::make_pair(masterSetOp.destPort(), maskValue));
              }
            }
      }
    }
    return portSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation* switchOp,
                           std::vector<PortMaskValue> nextPortMaskValues,
                           MaskValue maskValue) const {
    std::vector<PacketConnection> worklist;
    for (auto &nextPortMaskValue : nextPortMaskValues) {
      Port nextPort = nextPortMaskValue.first;
      MaskValue nextMaskValue = nextPortMaskValue.second;
      int maskConflicts = nextMaskValue.first & maskValue.first;
      //LLVM_DEBUG(llvm::dbgs() << "Mask: " << maskValue.first << " "
      //                        << maskValue.second << "\n");
      //LLVM_DEBUG(llvm::dbgs() << "NextMask: " << nextMaskValue.first << " "
      //                        << nextMaskValue.second << "\n");
      //LLVM_DEBUG(llvm::dbgs() << "conflicts: " << maskConflicts << "\n");

      if ((maskConflicts & nextMaskValue.second) !=
          (maskConflicts & maskValue.second)) {
        // Incoming packets cannot match this rule. Skip it.
        continue;
      }
      MaskValue newMaskValue = std::make_pair(
          maskValue.first | nextMaskValue.first,
          maskValue.second | (nextMaskValue.first & nextMaskValue.second));
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      // If there is no wire to follow then bail out.
      if (!nextConnection)
        continue;

      worklist.push_back(std::make_pair(*nextConnection, newMaskValue));
    }
    return worklist;
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection> getConnectedTiles(TileOp tileOp,
                                                  Port port) const {
    Coord src_coord = std::make_pair(tileOp.colIndex(), tileOp.rowIndex());
    LLVM_DEBUG(llvm::dbgs()
               << "\n\tBEGIN getConnectedTiles(" 
               << " Tile(" << src_coord.first << ", " << src_coord.second << "), "
               << stringifyWireBundle(port.first) << " " << (int)port.second << ")\n");

    // The accumulated result;
    std::vector<PacketConnection> connectedTiles;
    // A worklist of PortConnections to visit.  These are all input ports of
    // some object (likely either a TileOp or a SwitchboxOp).
    std::vector<PacketConnection> worklist;
    // Start the worklist by traversing from the tile to its connected
    // switchbox.
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);

    // If there is no wire to traverse, then just return no connection
    if (!t)
      return connectedTiles;
    worklist.push_back(std::make_pair(*t, std::make_pair(0, 0)));

    while (!worklist.empty()) {
      PacketConnection t = worklist.back();
      worklist.pop_back();
      PortConnection portConnection = t.first;
      MaskValue maskValue = t.second;
      Operation *other = portConnection.first;
      Port otherPort = portConnection.second;

      if (isa<FlowEndPoint>(other)) {
        // If we got to an endpoint tile, then add it to the result.
        LLVM_DEBUG(llvm::dbgs() << "Found FlowEndPoint:" << "\n");
        if (maskValue.first == 0) { // Circuit Switched flow
          connectedTiles.push_back(t);
        } else {// Packet flow
            LLVM_DEBUG(llvm::dbgs() << "else packet flow\n");
          if (TileOp endTileOp = dyn_cast_or_null<TileOp>(other)) {
            Coord coord = std::make_pair(endTileOp.colIndex(), endTileOp.rowIndex());
            SmallVector<PortID, 8>* pkt_ids = delivered_pkt_flow_ids.at(coord);
            LLVM_DEBUG(llvm::dbgs() << "(" << coord.first << ", " << coord.second << ") pkt_ids:\n");
            for ( PortID p : *pkt_ids) {
              printPortID(p);
              if(p.second == maskValue.second)
                connectedTiles.push_back(t);
            }
            LLVM_DEBUG(llvm::dbgs() << "\n");
          }
          else 
            LLVM_DEBUG(llvm::dbgs() << "No tileOp\n");
        }
      } else 
      if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp, switchOp.getConnections(),
                                           otherPort);
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } 
      else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp, switchOp.getConnections(),
                                           otherPort);
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      }
      else {
        LLVM_DEBUG(llvm::dbgs()
                   << "*** Connection Terminated at unknown operation: ");
        LLVM_DEBUG(other->dump());
      }
    }
    LLVM_DEBUG(llvm::dbgs() 
              << "\tEND getConnectedTile(): " 
              << connectedTiles.size() << " connected tiles found.\n");
    return connectedTiles;
  }
};

static void findFlowsFrom(AIE::TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "\n***Begin findFlowsFrom() tile ("
      << op.colIndex() << ", " << op.rowIndex() << ")\n");

  Operation *Op = op.getOperation();
  rewriter.setInsertionPointToEnd(Op->getBlock());

  std::pair<int, int> sb_coord = std::make_pair(op.colIndex(), op.rowIndex());

  std::vector<WireBundle> bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    for (int i = 0; i < op.getNumSourceConnections(bundle); i++) {
      // Assume any ID 0-31 can be sent from the start of a flow.
      // Therefore we initialize in_transit_pkt_flow_ids with all possible IDs.
      // We are checking for which IDs (if any) will reach a destination 
      // after following the packet routing rules.
      analysis.resetInTransitIDs();
      analysis.resetDeliveredIDs();
      for( unsigned int ID = 0; ID < 32; ID++){
          analysis.in_transit_pkt_flow_ids.at(sb_coord)->push_back(ID);
      }

      std::vector<PacketConnection> tiles =
          analysis.getConnectedTiles(op, std::make_pair(bundle, i));
        
      if(tiles.size() > 0)
        LLVM_DEBUG(llvm::dbgs() << "\n&&& " << tiles.size() << " tile(s) found connected to source tile (" << op.colIndex() << ", " << op.rowIndex() << ") " 
                              << stringifyWireBundle(bundle) << " : " << i << "\n");
      for(unsigned int t = 0; t < tiles.size(); t++) {
        TileOp tileOp = cast<TileOp>(tiles[t].first.first);
        LLVM_DEBUG(llvm::dbgs() << "Ends at: (" << tileOp.colIndex() << ", " << tileOp.rowIndex() << ")\n");
      }

      for (PacketConnection &c : tiles) {
        PortConnection portConnection = c.first;
        MaskValue maskValue = c.second;
        int mask = maskValue.first;
        int match = maskValue.second;
        Operation *destOp = portConnection.first;
        Port destPort = portConnection.second;
        TileOp destTileOp = cast<TileOp>(*destOp);
        std::pair<int, int> destCoord = 
            std::make_pair(destTileOp.colIndex(), destTileOp.rowIndex());

        // Circuit Switched flow
        if (maskValue.first == 0) {
          rewriter.create<FlowOp>(Op->getLoc(), Op->getResult(0), bundle, i,
                                  destOp->getResult(0), destPort.first,
                                  destPort.second);
        } 
        // Packet flow
        else {
          for(PortID p : *analysis.delivered_pkt_flow_ids.at(destCoord)) {
            Port delivered_port = p.first;
            int ID = p.second;
            if (delivered_port == destPort && ((ID & mask) == match)) {
              LLVM_DEBUG(llvm::dbgs() << "Creating new PacketFlowOp!\n");
              LLVM_DEBUG(llvm::dbgs() << "Src: ");
              LLVM_DEBUG(Op->dump());
              LLVM_DEBUG(llvm::dbgs() << "(" << stringifyWireBundle(bundle) << ":"
                          << i << ")\n");
              LLVM_DEBUG(llvm::dbgs() << "Dest: ");
              LLVM_DEBUG(destOp->dump());
              LLVM_DEBUG(llvm::dbgs() << "(" << stringifyWireBundle(destPort.first) << ":"
                          << destPort.second << ")\n");
              int flow_id = p.second;
              LLVM_DEBUG(llvm::dbgs() << "ID: " << flow_id << "\n");

              PacketFlowOp flowOp =
                  rewriter.create<PacketFlowOp>(Op->getLoc(), flow_id);
              flowOp.ensureTerminator(flowOp.getPorts(), rewriter, Op->getLoc());
              OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
              rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
              rewriter.create<PacketSourceOp>(Op->getLoc(), Op->getResult(0),
                                              bundle, (int)i);
              rewriter.create<PacketDestOp>(Op->getLoc(), destOp->getResult(0),
                                            destPort.first, (int)destPort.second);
              rewriter.restoreInsertionPoint(ip);
              LLVM_DEBUG(llvm::dbgs() << "Done creating PacketFlowOp!\n\n");
            }
          }
        }
      }
    }
  }
}

struct AIEFindFlowsPass : public AIEFindFlowsBase<AIEFindFlowsPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Begin AIEFindFlows...\n");
    ModuleOp m = getOperation();
    ConnectivityAnalysis analysis(m);
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    LLVM_DEBUG(llvm::dbgs() << "Begin processing tiles...\n");
    for (auto tile : m.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder);
    }
    analysis.print_delivered_pkt_flow_ids();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}