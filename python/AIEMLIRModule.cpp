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
         size_t partitionStartCol, bool enableCores) {
        mlir::python::CollectDiagnosticsToStringScope scope(
            mlirOperationGetContext(op));
        if (mlirLogicalResultIsFailure(aieTranslateToCDODirect(
                op, {workDirPath.data(), workDirPath.size()}, bigendian,
                emitUnified, cdoDebug, aieSim, xaieDebug, partitionStartCol,
                enableCores)))
          throw py::value_error("Failed to generate cdo because: " +
                                scope.takeMessage());
      },
      "module"_a, "work_dir_path"_a, "bigendian"_a = false,
      "emit_unified"_a = false, "cdo_debug"_a = false, "aiesim"_a = false,
      "xaie_debug"_a = false, "partition_start_col"_a = 1,
      "enable_cores"_a = true);

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
}
