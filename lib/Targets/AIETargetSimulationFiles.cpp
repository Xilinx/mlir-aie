//===- AIETargetSimulationFiles.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "aie/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "AIETargets.h"
namespace xilinx {
namespace AIE {

mlir::LogicalResult AIETranslateSCSimConfig(mlir::ModuleOp module,
                                            llvm::raw_ostream &output) {
  DeviceOp targetOp;
  for (auto tOp : module.getOps<DeviceOp>()) {
    targetOp = tOp;
    break; // Should only have 1 object in iterator
  }
  AIEArch arch = AIEArch::AIE1;
  if (targetOp) {
    arch = targetOp.getTargetModel().getTargetArch();
  }

  if (arch == AIEArch::AIE2) {
    output
        << "{\n"
        << "    \"SimulationConfig\": {\n"
        << "        \"device_json\": {\n"
        << "            \"directory\": \"data/aie_ml/devices\",\n"
        << "            \"file\": \"VC2802.json\"\n"
        << "        },\n"
        << "        \"phy_device_file\": \"xcve2802-vsvh1760-2LP-e-S-es1\",\n"
        << "        \"aiearch\": \"aie2\",\n"
        << "        \"aie_freq\": 1050000000.0,\n"
        << "        \"use_real_noc\": 1,\n"
        << "        \"evaluate_fifo_depth\": 0,\n"
        << "        \"noc_ip_block\": {\n"
        << "            \"lib_path\": \"./sim/noc/liblnoc_tlm.so\",\n"
        << "            \"traffic_file\": \"./sim/noc/noc_traffic.nts\",\n"
        << "            \"config_file\": \"./sim/noc/noc_soln.ncr\"\n"
        << "        },\n"
        << "        \"pl_ip_block\": [\n"
        << "            {\n"
        << "                \"name\": \"ps_ps_main\",\n"
        << "                \"ip\": \"ps\",\n"
        << "                \"lib_path\": \"ps/ps.so\",\n"
        << "                \"pl_freq\": 362500000.0,\n"
        << "                \"axi_mm\": [\n"
        << "                    {\n"
        << "                        \"port_name\": \"ps_axi\",\n"
        << "                        \"direction\": \"ps_to_gm\",\n"
        << "                        \"noc_endpoint\": \"NOC_NMU128_X0Y5\",\n"
        << "                        \"bus_width\": 0\n"
        << "                    }\n"
        << "                ],\n"
        << "                \"event_bus\": []\n"
        << "            }\n"
        << "        ]\n"
        << "    }\n"
        << "}\n";
    // AIE2 - ve2302
    // output << "{\n"
    //        << "    \"SimulationConfig\": {\n"
    //        << "        \"device_json\": {\n"
    //        << "            \"directory\": \"data/aie_ml/devices\",\n"
    //        << "            \"file\": \"VE2302.json\"\n"
    //        << "        },\n"
    //        << "        \"phy_device_file\":
    //        \"xcve2302-sfva784-1MP-e-S-es1\",\n"
    //        << "        \"aiearch\": \"aie2\",\n"
    //        << "        \"aie_freq\": 1150000000.0,\n"
    //        << "        \"use_real_noc\": 1,\n"
    //        << "        \"evaluate_fifo_depth\": 0,\n"
    //        << "        \"noc_ip_block\": {\n"
    //        << "            \"lib_path\": \"./sim/noc/liblnoc_tlm.so\",\n"
    //        << "            \"traffic_file\":
    //        \"./sim/noc/noc_traffic.nts\",\n"
    //        << "            \"config_file\": \"./sim/noc/noc_soln.ncr\"\n"
    //        << "        },\n"
    //        << "        \"pl_ip_block\": [\n"
    //        << "            {\n"
    //        << "                \"name\": \"ps_ps_main\",\n"
    //        << "                \"ip\": \"ps\",\n"
    //        << "                \"lib_path\": \"ps/ps.so\",\n"
    //        << "                \"pl_freq\": 312500000.0,\n"
    //        << "                \"axi_mm\": [\n"
    //        << "                    {\n"
    //        << "                        \"port_name\": \"ps_axi\",\n"
    //        << "                        \"direction\": \"ps_to_gm\",\n"
    //        << "                        \"noc_endpoint\":
    //        \"NOC_NMU128_X0Y5\",\n"
    //        << "                        \"bus_width\": 0\n"
    //        << "                    }\n"
    //        << "                ],\n"
    //        << "                \"event_bus\": []\n"
    //        << "            }\n"
    //        << "        ]\n"
    //        << "    }\n"
    //        << "}\n";
  } else { // AIEArch::AIE1
    output << "{\n"
           << "    \"SimulationConfig\": {\n"
           << "        \"device_json\": {\n"
           << "            \"directory\": \"data/devices\",\n"
           << "            \"file\": \"VC1902.json\"\n"
           << "        },\n"
           << "        \"phy_device_file\": \"xcvc1902-vsva2197-2MP-e-S\",\n"
           << "        \"aiearch\": \"aie\",\n"
           << "        \"aie_freq\": 1250000000.0,\n"
           << "        \"use_real_noc\": 1,\n"
           << "        \"evaluate_fifo_depth\": 0,\n"
           << "        \"noc_ip_block\": {\n"
           << "            \"lib_path\": \"./Work/noc/liblnoc_tlm.so\",\n"
           << "            \"traffic_file\": \"./Work/noc/noc_traffic.nts\",\n"
           << "            \"config_file\": \"./Work/noc/noc_soln.ncr\"\n"
           << "        },\n"
           << "        \"pl_ip_block\": [\n"
           << "            {\n"
           << "                \"name\": \"ps_ps_main\",\n"
           << "                \"ip\": \"ps\",\n"
           << "                \"lib_path\": \"ps/ps.so\",\n"
           << "                \"pl_freq\": 312500000.0,\n"
           << "                \"axi_mm\": [\n"
           << "                    {\n"
           << "                        \"port_name\": \"ps_axi\",\n"
           << "                        \"direction\": \"ps_to_gm\",\n"
           << "                        \"noc_endpoint\": \"NOC_NMU128_X0Y5\",\n"
           << "                        \"bus_width\": 0\n"
           << "                    }\n"
           << "                ],\n"
           << "                \"event_bus\": []\n"
           << "            }\n"
           << "        ]\n"
           << "    }\n"
           << "}\n";
  }
  return mlir::success();
}

/* Generates a aieshim_solution.aiesol file which is necessary to run aiesim.
Sample invocation:
aie-translate --aie-mlir-to-shim-solution ./aie.mlir >
./Work/arch/aieshim_solution.aiesol

NOTE: to correctly enable all tiles used for routing, the aie-opt routing pass
must be called first. So, a more practical invocation: aie-opt
--aie-create-pathfinder-flows ./aie.mlir | aie-translate --aie-mlir-to-shim >
./Work/arch/aieshim_solution.aiesol
*/
mlir::LogicalResult AIETranslateShimSolution(mlir::ModuleOp module,
                                             llvm::raw_ostream &output) {
  DeviceOp targetOp;
  for (auto tOp : module.getOps<DeviceOp>()) {
    targetOp = tOp;
    break; // Should only have 1 object in iterator
  }

  // Generate boilerplate header
  output << "{\n";
  output << "  \"Placement\": [\n";

  int shim_MM2S_count = 0;

  // For each DMAStartOp in shims, generate a "LogicalInstance" section
  auto all_shim_ops = targetOp.getOps<ShimDMAOp>();
  for (ShimDMAOp shimOp : all_shim_ops) {
    for (DMAStartOp startOp : shimOp.getOps<DMAStartOp>()) {
      // For aiesimulator to run, PortName must start at 00 and increase
      if (startOp.getChannelDir() == DMAChannelDir::MM2S) {
        if (shim_MM2S_count > 0)
          output << ",\n";

        std::string port_name = "";
        port_name.append("M");
        port_name.append(shim_MM2S_count < 10 ? "0" : ""); // padding zero
        port_name.append(std::to_string(shim_MM2S_count++));
        port_name.append("_AXI");
        // TODO: How to tell if PortName should be AXI or AXIS?

        // Generate a Logical Instance line
        output << "    {\n"
               << "      \"LogicalInstance\" : { \"InstanceName\" : "
               << "\"aie_engine_0\", \"PortName\" : \"" << port_name
               << "\"},\n";

        std::string col = std::to_string(shimOp.colIndex());
        int ch = startOp.getChannelIndex();
        std::string channel = std::to_string(ch);
        // "name" field appears to be arbitrary, but we try to be descriptive
        std::string physical_name = "";
        physical_name.append("AIE_NOC_X").append(col).append("Y0_AIE_NOC_");
        physical_name.append("M_AXI_ch").append(channel);

        // Generate a Physical Instance line
        output << "      \"PhysicalInstance\" : [{ \"name\" : \""
               << physical_name << "\", \"column\" : " << col
               << ", \"channel\" : " << channel << " }],\n"
               << "      \"IsSoft\" : true\n    }";
      }
    }
  }

  output << "\n  ]\n";
  output << "}\n";

  return mlir::success();
}

mlir::LogicalResult AIETranslateGraphXPE(mlir::ModuleOp module,
                                         llvm::raw_ostream &output) {
  /* Generates a .xpe file which is necessary to run aiesim.
  .xpe is a power report file, but has information on which AIE tiles are used.
  Sample invocation:
  aie-translate --aie-mlir-to-xpe ./aie.mlir > ./Work/reports/graph.xpe

  NOTE: to correctly enable all tiles used for routing, the aie-opt routing pass
  must be called first. So, a more practical invocation: aie-opt
  --aie-create-pathfinder-flows ./aie.mlir | aie-translate --aie-mlir-to-xpe >
  ./Work/reports/graph.xpe
  */

  DeviceOp targetOp;
  for (auto tOp : module.getOps<DeviceOp>()) {
    targetOp = tOp;
    break; // Should only have 1 object in iterator
  }
  AIEArch arch = AIEArch::AIE1;
  if (targetOp) {
    arch = targetOp.getTargetModel().getTargetArch();
  }

  // Generate boilerplate header
  // TODO: date and version should probably not be hardcoded
  output << "<?xml version=\"1.0\"?>"
         << "\n";
  output << "<POWERDATA data=\"AI-Engine Compiler\" dataVersion=\"2022.2\" "
            "design=\"graph\" date=\"2023\">\n";
  if (arch == AIEArch::AIE2) {
    output
        // AIE2 xcve2802
        << " <DEVICE part=\"xcve2802\" grade=\"extended\" package=\"vsvh1760\" "
           "speed=\"-2LP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
    // AIE2 - xcve2302
    // << " <DEVICE part=\"xcve2302\" grade=\"extended\" package=\"sfva784\" "
    //    "speed=\"-1MP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
  } else { // AIEArch::AIE1
    output
        << " <DEVICE part=\"xcvc1902\" grade=\"extended\" package=\"vsva2197\" "
           "speed=\"-2MP\" process=\"typical\" vid=\"No\"></DEVICE>\n";
  }
  output << "  <AIE status=\"COMPILER_OUTPUT\">\n";

  // Generate design specific info on tiles within the mlir module
  auto module_tile_ops = targetOp.getOps<TileOp>();
  int num_tiles = std::distance(module_tile_ops.begin(), module_tile_ops.end());
  // TODO: clk_freq only 1150 for AIE2
  if (arch == AIEArch::AIE2) {
    output << "    <AIE_MODULE name=\"graph\" num_tiles=\""
           << std::to_string(num_tiles) << "\" clk_freq=\"1150\">\n";
  } else {
    output << "    <AIE_MODULE name=\"graph\" num_tiles=\""
           << std::to_string(num_tiles) << "\" clk_freq=\"1250\">\n";
  }

  // Get all CoreOps into a convenient map which can then be referenced by
  // coordinates
  std::map<std::pair<int, int>, std::vector<CoreOp>> coreMap;
  for (CoreOp coreOp : targetOp.getOps<CoreOp>())
    coreMap[std::make_pair(coreOp.colIndex(), coreOp.rowIndex())].push_back(
        coreOp);

  // For each TileOp in the module, generate a <TILE> section
  int kernel_count = 0;
  for (TileOp tileOp : module_tile_ops) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    if (row == 0)
      continue; // Skip shim tiles (handled below)

    if (arch == AIEArch::AIE2) {
      output << "      <TILE name=\"CR(" <<
          // CR coordinates ignores shim, and 2 mem rows hence row-3
          // AIE2 - xcve2802
          std::to_string(col) << "," << std::to_string(row - 3)
             << ")\" "
             // CR coordinates ignores shim and 1 mem row, hence row-2
             // AIE2 - xcve2302
             // std::to_string(col) << "," << std::to_string(row - 2) << ")\" "
             << "type=\"int16\" int_core_load=\"1.0\" fp_core_load=\"0\" "
             << "mem_banks=\"0\" mem_rw_rate=\"0.2\" stream_util=\"0.0\" "
                "coordinates=\""
             <<
          // TODO: where does number of mem_banks come from?? Hardcoded to 0 for
          // now
          // TODO: does stream_util matter for sim?
          std::to_string(col) << "," << std::to_string(row - 1) << "\">\n";

    } else {
      output << "      <TILE name=\"CR(" <<
          // CR coordinates ignores shim, hence row-1
          std::to_string(col) << "," << std::to_string(row - 1) << ")\" "
             << "type=\"int16\" int_core_load=\"1.0\" fp_core_load=\"0\" "
             << "mem_banks=\"0\" mem_rw_rate=\"0.2\" stream_util=\"0.0\" "
                "coordinates=\""
             <<
          // TODO: where does number of mem_banks come from?? Hardcoded to 0 for
          // now
          // TODO: does stream_util matter for sim?
          std::to_string(col) << "," << std::to_string(row) << "\">\n";
    }

    // If the TileOp has associated Kernels, generate <KERNEL> sections
    auto coreOps_in_tile =
        coreMap[std::make_pair(tileOp.colIndex(), tileOp.rowIndex())];
    for (auto coreOp : coreOps_in_tile) {
      (void)(coreOp); // get around unused variable warning for coreOp
      output << "        <KERNEL name=\"i" << std::to_string(kernel_count++)
             << "\" "
             << "int_core_load=\"" << std::to_string(1 / coreOps_in_tile.size())
             << "\" fp_core_load=\"0\"></KERNEL>\n";
    }
    output << "      </TILE>\n";
  }

  // For each ShimOp in the module, generate a <SHIM> section
  for (ShimDMAOp shimOp : targetOp.getOps<ShimDMAOp>()) {
    output << "      <SHIM name=\"SHIM(" << std::to_string(shimOp.colIndex())
           << ", " << std::to_string(shimOp.rowIndex()) << ")\" " <<
        // TODO: stream_util can be 0 for aiesim purposes?
        "type=\"AIE_PL_NOC_SHIM\" stream_util=\"0\" num_pl_streams=\"0\" " <<
        // TODO: how to get num_aximm_connections from mlir?
        "num_aximm_connections=\"1\" coordinates=\""
           << std::to_string(shimOp.colIndex()) << ","
           << std::to_string(shimOp.rowIndex()) << "\" "
           << "></SHIM>\n";
  }

  output << "    </AIE_MODULE>\n";
  output << "  </AIE>\n";
  output << "</POWERDATA>\n";

  return mlir::success();
}

} // namespace AIE
} // namespace xilinx