//===- AIEMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/Dialects.h"
#include "aie-c/Registration.h"
#include "aie-c/Translation.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unicodeobject.h>
#include <vector>

using namespace mlir::python::adaptors;
namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_aie, m) {

  aieRegisterAllPasses();

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) {
        MlirDialectHandle aieHandle = mlirGetDialectHandle__aie__();
        MlirDialectHandle aiexHandle = mlirGetDialectHandle__aiex__();
        MlirDialectHandle aievecHandle = mlirGetDialectHandle__aievec__();
        mlirDialectHandleInsertDialect(aieHandle, registry);
        mlirDialectHandleInsertDialect(aiexHandle, registry);
        mlirDialectHandleInsertDialect(aievecHandle, registry);
      },
      "registry"_a);

  // AIE types bindings
  mlir_type_subclass(m, "ObjectFifoType", aieTypeIsObjectFifoType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoTypeGet(type));
          },
          "Get an instance of ObjectFifoType with given element type.",
          "self"_a, "type"_a = py::none());

  mlir_type_subclass(m, "ObjectFifoSubviewType", aieTypeIsObjectFifoSubviewType)
      .def_classmethod(
          "get",
          [](const py::object &cls, const MlirType type) {
            return cls(aieObjectFifoSubviewTypeGet(type));
          },
          "Get an instance of ObjectFifoSubviewType with given element type.",
          "self"_a, "type"_a = py::none());

  auto stealCStr = [](MlirStringRef mlirString) {
    if (!mlirString.data || mlirString.length == 0)
      throw std::runtime_error("couldn't translate");
    std::string cpp(mlirString.data, mlirString.length);
    free((void *)mlirString.data);
    py::handle pyS = PyUnicode_DecodeLatin1(cpp.data(), cpp.length(), nullptr);
    if (!pyS)
      throw py::error_already_set();
    return py::reinterpret_steal<py::str>(pyS);
  };

  m.def(
      "translate_aie_vec_to_cpp",
      [&stealCStr](MlirOperation op, bool aieml) {
        return stealCStr(aieTranslateAIEVecToCpp(op, aieml));
      },
      "module"_a, "aieml"_a = false);

  m.def(
      "translate_mlir_to_llvmir",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateModuleToLLVMIR(op));
      },
      "module"_a);

  m.def(
      "generate_cdo",
      [](MlirOperation op, const std::string &workDirPath, bool bigendian,
         bool emitUnified, bool cdoDebug, bool aieSim, bool xaieDebug,
         bool enableCores) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(op));
        if (mlirLogicalResultIsFailure(aieTranslateToCDODirect(
                op, {workDirPath.data(), workDirPath.size()}, bigendian,
                emitUnified, cdoDebug, aieSim, xaieDebug, enableCores)))
          throw py::value_error("Failed to generate cdo because: " +
                                scope.takeMessage());
      },
      "module"_a, "work_dir_path"_a, "bigendian"_a = false,
      "emit_unified"_a = false, "cdo_debug"_a = false, "aiesim"_a = false,
      "xaie_debug"_a = false, "enable_cores"_a = true);

  m.def(
      "npu_instgen",
      [&stealCStr](MlirOperation op) {
        py::str npuInstructions = stealCStr(aieTranslateToNPU(op));
        auto individualInstructions =
            npuInstructions.attr("split")().cast<py::list>();
        for (size_t i = 0; i < individualInstructions.size(); ++i)
          individualInstructions[i] = individualInstructions[i].attr("strip")();
        return individualInstructions;
      },
      "module"_a);

  m.def(
      "generate_xaie",
      [&stealCStr](MlirOperation op) {
        return stealCStr(aieTranslateToXAIEV2(op));
      },
      "module"_a);

  m.def(
      "generate_bcf",
      [&stealCStr](MlirOperation op, int col, int row) {
        return stealCStr(aieTranslateToBCF(op, col, row));
      },
      "module"_a, "col"_a, "row"_a);

  m.def(
      "aie_llvm_link",
      [&stealCStr](std::vector<std::string> moduleStrs) {
        std::vector<MlirStringRef> modules;
        modules.reserve(moduleStrs.size());
        for (auto &moduleStr : moduleStrs)
          modules.push_back({moduleStr.data(), moduleStr.length()});

        return stealCStr(aieLLVMLink(modules.data(), modules.size()));
      },
      "modules"_a);

  m.def(
      "get_target_model",
      [](uint32_t d) -> const xilinx::AIE::AIETargetModel & {
        auto aiedev = static_cast<xilinx::AIE::AIEDevice>(d);
        return xilinx::AIE::getTargetModel(aiedev);
      },
      py::return_value_policy::reference);

  py::class_<xilinx::AIE::AIETargetModel>(m, "AIETargetModel")
      .def("columns", &xilinx::AIE::AIETargetModel::columns,
           py::return_value_policy::reference)
      .def("rows", &xilinx::AIE::AIETargetModel::rows,
           py::return_value_policy::reference)
      .def("is_core_tile", &xilinx::AIE::AIETargetModel::isCoreTile,
           py::return_value_policy::reference)
      .def("is_mem_tile", &xilinx::AIE::AIETargetModel::isMemTile,
           py::return_value_policy::reference)
      .def("is_shim_noc_tile", &xilinx::AIE::AIETargetModel::isShimNOCTile,
           py::return_value_policy::reference)
      .def("is_shim_pl_tile", &xilinx::AIE::AIETargetModel::isShimPLTile,
           py::return_value_policy::reference)
      .def("is_shim_noc_or_pl_tile",
           &xilinx::AIE::AIETargetModel::isShimNOCorPLTile,
           py::return_value_policy::reference)
      .def("is_valid_tile", &xilinx::AIE::AIETargetModel::isValidTile,
           py::return_value_policy::reference)
      .def("is_valid_trace_master",
           &xilinx::AIE::AIETargetModel::isValidTraceMaster,
           py::return_value_policy::reference)
      .def("get_local_memory_size",
           &xilinx::AIE::AIETargetModel::getLocalMemorySize,
           py::return_value_policy::reference)
      .def("get_num_mem_tile_rows",
           &xilinx::AIE::AIETargetModel::getNumMemTileRows,
           py::return_value_policy::reference)
      .def("get_mem_tile_size", &xilinx::AIE::AIETargetModel::getMemTileSize,
           py::return_value_policy::reference)
      .def("get_num_source_switchbox_connections",
           &xilinx::AIE::AIETargetModel::getNumSourceSwitchboxConnections,
           py::return_value_policy::reference)
      .def("get_num_dest_switchbox_connections",
           &xilinx::AIE::AIETargetModel::getNumDestSwitchboxConnections,
           py::return_value_policy::reference)
      .def("is_npu", &xilinx::AIE::AIETargetModel::isNPU,
           py::return_value_policy::reference);

  py::class_<xilinx::AIE::AIE1TargetModel, xilinx::AIE::AIETargetModel>(
      m, "AIE1TargetModel");
  py::class_<xilinx::AIE::VC1902TargetModel, xilinx::AIE::AIE1TargetModel>(
      m, "VC1902TargetModel")
      .def(py::init());

  py::class_<xilinx::AIE::AIE2TargetModel, xilinx::AIE::AIETargetModel>(
      m, "AIE2TargetModel");
  py::class_<xilinx::AIE::VE2302TargetModel, xilinx::AIE::AIE2TargetModel>(
      m, "VE2302TargetModel")
      .def(py::init());
  py::class_<xilinx::AIE::VE2802TargetModel, xilinx::AIE::AIE2TargetModel>(
      m, "VE2802TargetModel")
      .def(py::init());

  py::class_<xilinx::AIE::BaseNPUTargetModel, xilinx::AIE::AIE2TargetModel>(
      m, "BaseNPUTargetModel");
  py::class_<xilinx::AIE::NPUTargetModel, xilinx::AIE::BaseNPUTargetModel>(
      m, "NPUTargetModel")
      .def(py::init());
  py::class_<xilinx::AIE::VirtualizedNPUTargetModel,
             xilinx::AIE::BaseNPUTargetModel>(m, "VirtualizedNPUTargetModel")
      .def(py::init<int>());
}
