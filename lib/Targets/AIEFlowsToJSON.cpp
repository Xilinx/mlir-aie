//===- AIEFlowsToJSON.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

/*
* Takes as input the mlir after AIECreateFlows and AIEFindFlows.
* Converts the flows into a JSON file to be read by other tools.
*/

#include <queue>
#include <set>

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
			return WireBundle::South; break;
		case WireBundle::South:
			return WireBundle::North; break;
		case WireBundle::East :
			return WireBundle::West; break;
		case WireBundle::West :
			return WireBundle::East; break;
		default: return bundle;
	}
}

//returns coordinates in the direction indicated by bundle
std::pair<uint32_t, uint32_t> getNextCoords(
	uint32_t col, uint32_t row, WireBundle bundle) {
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

	// count flow sources and destinations
	std::map< TileID, int > source_counts;
	std::map< TileID, int > destination_counts;
	for(FlowOp flowOp : module.getOps<FlowOp>()) {
		TileOp source = cast<TileOp>(flowOp.source().getDefiningOp());
		TileOp dest = cast<TileOp>(flowOp.dest().getDefiningOp());
		TileID srcID = std::make_pair(source.colIndex(), source.rowIndex());
		TileID dstID = std::make_pair(dest.colIndex(), dest.rowIndex());
		source_counts[srcID]++;
		destination_counts[dstID]++;
	}

	// for each switchbox, write name, coordinates, and routing demand info
	for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
		uint32_t col = switchboxOp.colIndex();
		uint32_t row = switchboxOp.rowIndex();
		std::string switchString = "\"switchbox" + std::to_string(col) +
					std::to_string(row) + "\": {\n" + 
					"\"col\": " + std::to_string(col) + ",\n" +
					"\"row\": " + std::to_string(row) + ",\n" ;

		// write source and destination info
		switchString += "\"source_count\": " + std::to_string(
							source_counts[std::make_pair(col, row)]) + ",\n";
		switchString += "\"destination_count\": " + std::to_string(
							destination_counts[std::make_pair(col, row)]) + ",\n";
		
		// write routing demand info
		uint32_t connect_counts[10];
		for(int i = 0; i < 10; i++) connect_counts[i] = 0;
		for(ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) 
			connect_counts[int(connectOp.destBundle())]++;

		switchString += "\"northbound\": " + std::to_string(
					connect_counts[int(WireBundle::North)]) + ",\n";
		switchString += "\"eastbound\": "  + std::to_string(
					connect_counts[int(WireBundle::East)]) + ",\n";
		switchString += "\"southbound\": " + std::to_string(
					connect_counts[int(WireBundle::South)]) + ",\n";
		switchString += "\"westbound\": "  + std::to_string(
					connect_counts[int(WireBundle::West)]) + "\n";
		switchString += "},\n";
		output << switchString;
	}

	// for each flow, trace it through switchboxes and write the route to JSON 
	int flow_count = 0;
	std::set< std::pair<TileOp, Port> > flowSources;
	for(FlowOp flowOp : module.getOps<FlowOp>()) {
		// objects used to trace through the flow
		Port curr_port = std::make_pair(flowOp.sourceBundle(),
												  flowOp.sourceChannel());
		SwitchboxOp curr_switchbox;

		TileOp source = cast<TileOp>(flowOp.source().getDefiningOp());
		//TileOp dest = cast<TileOp>(flowOp.dest().getDefiningOp());

		//track flow sources to avoid duplicate routes
		auto flowSource = std::make_pair(source, curr_port);
		if(flowSources.count(flowSource)) {
			continue;
		}
		flowSources.insert(flowSource);
		
		std::string routeString = "\"route"+std::to_string(flow_count++)+"\": [ ";

		//FIFO to handle fanouts
		std::queue<Port> next_ports;
		std::queue<SwitchboxOp> next_switches;

		// find the starting switchbox
		for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
			if(switchboxOp.colIndex() == source.colIndex() &&
				 switchboxOp.rowIndex() == source.rowIndex()) {
				curr_switchbox = switchboxOp;
				break;
			}
		}
	
		//if the flow starts in a shim, handle seperately
		for(ShimMuxOp shimMuxOp: module.getOps<ShimMuxOp>()) {
			if(shimMuxOp.colIndex() == source.colIndex() &&
				 shimMuxOp.rowIndex() == source.rowIndex()) {
				for(ConnectOp connectOp : shimMuxOp.getOps<ConnectOp>()) {
					if(connectOp.sourceBundle() == curr_port.first && 
					connectOp.sourceChannel() == (unsigned)curr_port.second) {
						curr_port.first = getConnectingBundle(connectOp.destBundle()); 
						curr_port.second = connectOp.destChannel();
						break;
					}
				}
				break;
			}
		}

		// trace through the flow and add switchbox coordinates to JSON
		bool done = false;
		do {
			// get the coordinates for the next switchbox in the flow																					 
			for(ConnectOp connectOp : curr_switchbox.getOps<ConnectOp>()) {
				//if this connectOp is the end of a flow, skip
				if( (curr_switchbox.rowIndex() == 0 && connectOp.destBundle() == WireBundle::South) ||
				connectOp.destBundle() == WireBundle::DMA || connectOp.destBundle() == WireBundle::Core )
					continue;

				if(connectOp.sourceBundle() == curr_port.first && 
				connectOp.sourceChannel() == (unsigned)curr_port.second) {
					next_ports.push(std::make_pair(
						getConnectingBundle(connectOp.destBundle()), 
						connectOp.destChannel()));

					std::pair<uint32_t, uint32_t> next_coords = 
						getNextCoords(curr_switchbox.colIndex(),
						curr_switchbox.rowIndex(), connectOp.destBundle());

					// search for next switchbox to connect to
					for(SwitchboxOp switchboxOp : module.getOps<SwitchboxOp>()) {
						if(uint32_t(switchboxOp.colIndex()) == next_coords.first &&
							uint32_t(switchboxOp.rowIndex()) == next_coords.second) {
							next_switches.push(switchboxOp);
							break;
						}
					}
				}
			}

			// add switchbox to the routeString
			std::string dirString = std::string("[[") + 
				std::to_string(curr_switchbox.colIndex()) + ", " + 
				std::to_string(curr_switchbox.rowIndex()) + "], [";
			int op_count = 0, dir_count = 0;
			for(ConnectOp connectOp : curr_switchbox.getOps<ConnectOp>()) {
				if(connectOp.sourceBundle() == curr_port.first &&
				connectOp.sourceChannel() == (unsigned)curr_port.second) {
					if(op_count++ > 0) dirString += ", ";
					dir_count++;
					dirString += "\"" + 
					(std::string)stringifyWireBundle(connectOp.destBundle()) + "\"";			
				}
			}
			dirString += "]], ";
			if(dir_count > 0)
				routeString += dirString;

			// if in the same switchbox, ignore it
			//if(curr_switchbox == next_switches.front()) {
			//	next_ports.pop();
			//	next_switches.pop();
			//}
			if(next_ports.empty() || next_switches.empty()) {
				done = true;
				routeString += "[]";
			} else {
				curr_port = next_ports.front();
				curr_switchbox = next_switches.front();
				next_ports.pop();
				next_switches.pop();
			}
		} while(!done);
		// write string to JSON
		routeString += std::string(" ],\n");
		output << routeString;
	}
	output << "\n\"end json\": 0\n"; // dummy line to avoid errors from commas
	output << "}";
	return success();
} // end AIETranslateToJSON
} // end AIE	
} // end xilinx