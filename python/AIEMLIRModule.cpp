//===- AIEMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie-c/Dialects.h"
#include "aie-c/Registration.h"
#include "aie-c/TargetModel.h"
#include "aie-c/Translation.h"

#include "aie/Bindings/PyTypes.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

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
      [&stealCStr](MlirOperation op, bool aie2) {
        return stealCStr(aieTranslateAIEVecToCpp(op, aie2));
      },
      "module"_a, "aie2"_a = false);

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
      "generate_ctrlpkt",
      [](MlirOperation op, const std::string &outputFile,
         const std::string &workDirPath, bool aieSim, bool xaieDebug,
         bool enableCores) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(op));
        if (mlirLogicalResultIsFailure(aieTranslateToCtrlpkt(
                op, {outputFile.data(), outputFile.size()},
                {workDirPath.data(), workDirPath.size()}, aieSim, xaieDebug,
                enableCores)))
          throw py::value_error("Failed to generate control packets because: " +
                                scope.takeMessage());
      },
      "module"_a, "output_file"_a, "work_dir_path"_a, "aiesim"_a = false,
      "xaie_debug"_a = false, "enable_cores"_a = true);

  m.def(
      "transaction_binary_to_mlir",
      [](MlirContext ctx, py::bytes bytes) {
        std::string s = bytes;
        MlirStringRef bin = {s.data(), s.size()};
        return aieTranslateBinaryToTxn(ctx, bin);
      },
      "ctx"_a, "binary"_a);

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
      "generate_control_packets",
      [&stealCStr](MlirOperation op) {
        py::str ctrlPackets =
            stealCStr(AIETranslateControlPacketsToUI32Vec(op));
        auto individualInstructions =
            ctrlPackets.attr("split")().cast<py::list>();
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

  m.def("get_target_model",
        [](uint32_t d) -> PyAieTargetModel { return aieGetTargetModel(d); });

  py::class_<PyAieTargetModel>(m, "AIETargetModel", py::module_local())
      .def(
          "columns",
          [](PyAieTargetModel &self) {
            return aieTargetModelColumns(self.get());
          },
          "Get the number of columns in the device.")
      .def(
          "rows",
          [](PyAieTargetModel &self) { return aieTargetModelRows(self.get()); },
          "Get the number of rows in the device.")
      .def("is_core_tile",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelIsCoreTile(self.get(), col, row);
           })
      .def("is_mem_tile",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelIsMemTile(self.get(), col, row);
           })
      .def("is_shim_noc_tile",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelIsShimNOCTile(self.get(), col, row);
           })
      .def("is_shim_pl_tile",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelIsShimPLTile(self.get(), col, row);
           })
      .def("is_shim_noc_or_pl_tile",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelIsShimNOCorPLTile(self.get(), col, row);
           })
      // .def("is_valid_tile")
      // .def("is_valid_trace_master")
      // .def("get_mem_west")
      // .def("get_mem_east")
      // .def("get_mem_north")
      // .def("get_mem_south")
      .def("is_internal",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsInternal(self.get(), src_col, src_row,
                                             dst_col, dst_row);
           })
      .def("is_west",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsWest(self.get(), src_col, src_row, dst_col,
                                         dst_row);
           })
      .def("is_east",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsEast(self.get(), src_col, src_row, dst_col,
                                         dst_row);
           })
      .def("is_north",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsNorth(self.get(), src_col, src_row, dst_col,
                                          dst_row);
           })
      .def("is_south",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsSouth(self.get(), src_col, src_row, dst_col,
                                          dst_row);
           })
      .def("is_mem_west",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsMemWest(self.get(), src_col, src_row,
                                            dst_col, dst_row);
           })
      .def("is_mem_east",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsMemEast(self.get(), src_col, src_row,
                                            dst_col, dst_row);
           })
      .def("is_mem_north",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsMemNorth(self.get(), src_col, src_row,
                                             dst_col, dst_row);
           })
      .def("is_mem_south",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsMemSouth(self.get(), src_col, src_row,
                                             dst_col, dst_row);
           })
      .def("is_legal_mem_affinity",
           [](PyAieTargetModel &self, int src_col, int src_row, int dst_col,
              int dst_row) {
             return aieTargetModelIsLegalMemAffinity(self.get(), src_col,
                                                     src_row, dst_col, dst_row);
           })
      //.def("get_mem_internal_base_address")
      .def("get_mem_west_base_address",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetMemWestBaseAddress(self.get());
           })
      .def("get_mem_east_base_address",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetMemEastBaseAddress(self.get());
           })
      .def("get_mem_north_base_address",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetMemNorthBaseAddress(self.get());
           })
      .def("get_mem_south_base_address",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetMemSouthBaseAddress(self.get());
           })
      .def("get_local_memory_size",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetLocalMemorySize(self.get());
           })
      .def("get_num_locks",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelGetNumLocks(self.get(), col, row);
           })
      .def("get_num_bds",
           [](PyAieTargetModel &self, int col, int row) {
             return aieTargetModelGetNumBDs(self.get(), col, row);
           })
      .def("get_num_mem_tile_rows",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetNumMemTileRows(self.get());
           })
      .def("get_mem_tile_size",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetMemTileSize(self.get());
           })
      // .def("get_num_dest_switchbox_connections", int col, int row)
      // .def("get_num_source_switchbox_connections", int col, int row)
      // .def("get_num_dest_shim_mux_connections", int col, int row)
      // .def("get_num_source_shim_mux_connections", int col, int row)
      // .def("is_legal_memtile_connection")
      .def("is_npu",
           [](PyAieTargetModel &self) {
             return aieTargetModelIsNPU(self.get());
           })
      .def("get_column_shift",
           [](PyAieTargetModel &self) {
             return aieTargetModelGetColumnShift(self.get());
           })
      .def("get_row_shift", [](PyAieTargetModel &self) {
        return aieTargetModelGetRowShift(self.get());
      });
}
