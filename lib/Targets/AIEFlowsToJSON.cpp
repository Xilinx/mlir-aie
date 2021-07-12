//===- AIEFlowsToJSON.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
* AIETranslateToJSON.cpp
* Takes as input the mlir after AIECreateFlows and AIEFindFlows.
* Converts the flows into a JSON file to be read by other tools.
*/

#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "AIEDialect.h"
#include "AIENetlistAnalysis.h"

#include "AIETargets.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx {
namespace AIE {
		

//returns the opposite WireBundle, useful for connecting switchboxes
WireBundle getConnectingBundle(WireBundle bundle) {
	switch(bundle) {
		case WireBundle::North :
			return WireBundle::South;
		case WireBundle::South:
			return WireBundle::North;
		case WireBundle::East :
			return WireBundle::West;
		case WireBundle::West :
			return WireBundle::East;
		default: return bundle;
	}
}

//returns coordinates in the direction indicated by bundle
std::pair<uint32_t, uint32_t> getNextCoords(uint32_t col, uint32_t row, WireBundle bundle) {
	switch(bundle) {
		case WireBundle::North :
			return std::make_pair(col, row+1);
		case WireBundle::South:
			return std::make_pair(col, row-1);
		case WireBundle::East :
			return std::make_pair(col+1, row);
		case WireBundle::West :
			return std::make_pair(col-1, row);
		default:
			return std::make_pair(col, row);
	}
}

mlir::LogicalResult AIEFlowsToJSON(ModuleOp module, raw_ostream &output) {
	output << "{\n";

	// for each switchbox, write name, coordinates, and routing demand info
	for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
		uint32_t col = switchboxOp.colIndex();
		uint32_t row = switchboxOp.rowIndex();
		std::string switchString = "\"switchbox" + std::to_string(col) +
					std::to_string(row) + "\": {\n" + 
					"\"col\": " + std::to_string(col) + ",\n" +
					"\"row\": " + std::to_string(row) + ",\n" ;
		
		// write routing demand info
		uint32_t connect_counts[10];
		for(int i = 0; i < 10; i++) connect_counts[i] = 0;
		for(ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) 
			connect_counts[int(connectOp.destBundle())]++;

		switchString += "\"northbound\": " + std::to_string(connect_counts[int(WireBundle::North)]) + ",\n";
		switchString += "\"eastbound\": "  + std::to_string(connect_counts[int(WireBundle::East)]) + ",\n";
		switchString += "\"southbound\": " + std::to_string(connect_counts[int(WireBundle::South)]) + ",\n";
		switchString += "\"westbound\": "  + std::to_string(connect_counts[int(WireBundle::West)]) + "\n";
		switchString += "},\n";
		output << switchString;
	}

	// for each flow, trace it through switchboxes and write the route to JSON 
	int flow_count = 0;
	for(FlowOp flowOp : module.getOps<FlowOp>()) {
		TileOp source = cast<TileOp>(flowOp.source().getDefiningOp());
		TileOp dest = cast<TileOp>(flowOp.dest().getDefiningOp());

		std::string routeString = "\"route" + std::to_string(flow_count++) + "\": [";

		// objects used to trace through the flow
		WireBundle curr_bundle = flowOp.sourceBundle();
		uint32_t curr_channel = flowOp.sourceChannel();
		SwitchboxOp curr_switchbox;
		WireBundle next_bundle;
		uint32_t next_channel;

		// find the starting switchbox
		for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
			if(switchboxOp.colIndex() == source.colIndex() &&
				 switchboxOp.rowIndex() == source.rowIndex()) {
				curr_switchbox = switchboxOp;
				break;
			}
		}
	
		//if the flow starts in a shim, handle seperately
		ShimMuxOp curr_shim;
		for(ShimMuxOp shimMuxOp: module.getOps<ShimMuxOp>()) {
			if(shimMuxOp.colIndex() == source.colIndex() &&
				 shimMuxOp.rowIndex() == source.rowIndex()) {
				curr_shim = shimMuxOp;
				for(ConnectOp connectOp : curr_shim.getOps<ConnectOp>()) {
					if(connectOp.sourceBundle() == curr_bundle && connectOp.sourceChannel() == curr_channel) {
						curr_bundle = getConnectingBundle(connectOp.destBundle());
						curr_channel = connectOp.destChannel(); 
						break;
					}
				}
				break;
			}
		}

		// trace through the flow and add switchbox coordinates to JSON
		bool done = false;
		while(!done) {
			// get the coordinates for the next switchbox in the flow																					 
			bool next_found = false;										
			for(ConnectOp connectOp : curr_switchbox.getOps<ConnectOp>()) {
				if(connectOp.sourceBundle() == curr_bundle && connectOp.sourceChannel() == curr_channel) {
					next_bundle = connectOp.destBundle();
					next_channel = connectOp.destChannel(); 
					next_found = true;
					break;
				}
			}

			if(!next_found) {
				llvm::dbgs() << "\tWARNING: Incomplete flow detected!\n" <<
									 "\tFrom: " << *source << "\n" <<
									 "\tTo: " << *dest << "\n";
				routeString += std::string("[") + std::to_string(curr_switchbox.colIndex()) + ", " + 
																					std::to_string(curr_switchbox.rowIndex()) + "]],\n";
				output << routeString;
				break;
			}
			// add switchbox to the routeString
			routeString += std::string("[") + std::to_string(curr_switchbox.colIndex()) + ", " + 
																				std::to_string(curr_switchbox.rowIndex()) + "], ";

			std::pair<uint32_t, uint32_t> next_coords = getNextCoords(curr_switchbox.colIndex(),
										 curr_switchbox.rowIndex(), next_bundle);
			// search for next switchbox to connect to
			for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
				if(uint32_t(switchboxOp.colIndex()) == next_coords.first &&
					 uint32_t(switchboxOp.rowIndex()) == next_coords.second) {
					curr_switchbox = switchboxOp;
					break;
				}
			}
			curr_bundle = getConnectingBundle(next_bundle);
			curr_channel = next_channel;

			// check if destination has been reached
			if(curr_switchbox.colIndex() == dest.colIndex() && curr_switchbox.rowIndex() == dest.rowIndex()) {
				done = true;
				// write the destination switchbox to routeString
					routeString += std::string("[") + std::to_string(curr_switchbox.colIndex()) + ", " + 
																					 std::to_string(curr_switchbox.rowIndex()) + "]],\n";
				// write string to JSON
				output << routeString;
			}
		}
	}
	output << "\"\": 1\n"; // dummy line to avoid errors from commas
	output << "}";
	return success();
} // end AIETranslateToJSON
} // end AIE	
} // end xilinx