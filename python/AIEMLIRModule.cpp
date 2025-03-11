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
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "llvm/ADT/Twine.h"

#include <nanobind/nanobind.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unicodeobject.h>
#include <vector>

using namespace mlir::python;
namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_aie, m) {

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
  nanobind_adaptors::mlir_type_subclass(m, "ObjectFifoType",
                                        aieTypeIsObjectFifoType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const MlirType type) {
            return cls(aieObjectFifoTypeGet(type));
          },
          "Get an instance of ObjectFifoType with given element type.",
          "self"_a, "type"_a = nb::none());

  nanobind_adaptors::mlir_type_subclass(m, "ObjectFifoSubviewType",
                                        aieTypeIsObjectFifoSubviewType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const MlirType type) {
            return cls(aieObjectFifoSubviewTypeGet(type));
          },
          "Get an instance of ObjectFifoSubviewType with given element type.",
          "self"_a, "type"_a = nb::none());

  auto stealCStr = [](MlirStringRef mlirString) {
    if (!mlirString.data || mlirString.length == 0)
      throw std::runtime_error("couldn't translate");
    std::string cpp(mlirString.data, mlirString.length);
    free((void *)mlirString.data);
    nb::handle pyS = PyUnicode_DecodeLatin1(cpp.data(), cpp.length(), nullptr);
    if (!pyS)
      throw nb::python_error();
    return nb::steal<nb::str>(pyS);
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
          throw nb::value_error(
              (llvm::Twine("Failed to generate cdo because: ") +
               llvm::Twine(scope.takeMessage()))
                  .str()
                  .c_str());
      },
      "module"_a, "work_dir_path"_a, "bigendian"_a = false,
      "emit_unified"_a = false, "cdo_debug"_a = false, "aiesim"_a = false,
      "xaie_debug"_a = false, "enable_cores"_a = true);

  m.def(
      "transaction_binary_to_mlir",
      [](MlirContext ctx, nb::bytes bytes) {
        MlirStringRef bin = {static_cast<const char *>(bytes.data()),
                             bytes.size()};
        return aieTranslateBinaryToTxn(ctx, bin);
      },
      "ctx"_a, "binary"_a);

  m.def(
      "translate_npu_to_binary",
      [](MlirOperation op, const std::string &sequence_name) {
        MlirStringRef instStr = aieTranslateNpuToBinary(
            op, {sequence_name.data(), sequence_name.size()});
        std::vector<uint32_t> vec(
            reinterpret_cast<const uint32_t *>(instStr.data),
            reinterpret_cast<const uint32_t *>(instStr.data) + instStr.length);
        free((void *)instStr.data);
        return vec;
      },
      "module"_a, "sequence_name"_a = "");

  m.def(
      "generate_control_packets",
      [](MlirOperation op) {
        MlirStringRef instStr = aieTranslateControlPacketsToUI32Vec(op);
        std::vector<uint32_t> vec(
            reinterpret_cast<const uint32_t *>(instStr.data),
            reinterpret_cast<const uint32_t *>(instStr.data) + instStr.length);
        free((void *)instStr.data);
        return vec;
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

  m.def("runtime_sequence_create", [](const std::string &name, int device) {
    return aieRuntimeSequenceCreate({name.data(), name.size()}, device);
  });

  m.def("runtime_sequence_add_dma_memcpy",
        [&stealCStr](MlirOperation runtime_sequence, uint32_t direction, uint32_t id,
           uint32_t channel, uint32_t column, uint64_t addr,
           std::vector<uint32_t> offsets, std::vector<uint32_t> sizes,
           std::vector<uint32_t> strides) {
            return stealCStr(aieRuntimeSequenceAddNpuDmaMempy(
              runtime_sequence, id, direction, channel, column, addr,
              offsets.data(), sizes.data(), strides.data()));
        });

  m.def("runtime_sequence_add_dma_wait",
        [](MlirOperation runtime_sequence, const std::string &symbol) {
          return aieRuntimeSequenceAddNpuDmaWait(
              runtime_sequence, {symbol.data(), symbol.size()}).value;
        });

  nb::class_<PyAieTargetModel>(m, "AIETargetModel")
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
