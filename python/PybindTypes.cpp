//===- PybindTypes.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PybindTypes.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/operators.h>

using namespace mlir;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::set<Port>);

void xilinx::AIE::bindTypes(py::module_ &m) {
  // By default, stl containers aren't passed by reference but passed by value,
  // so they can't be mutated.
  py::class_<std::set<Port>>(m, "PortSet")
      .def(py::init<>())
      .def("add", [](std::set<Port> &set, const Port p) { set.insert(p); })
      .def("__repr__",
           [](const std::set<Port> &set) {
             std::stringstream ss;
             ss << "{"
                << join(llvm::map_range(
                            set,
                            [](const Port &port) { return to_string(port); }),
                        ", ")
                << "}";

             return ss.str();
           })
      .def("__contains__", [](const std::set<Port> &self, const Port &p) {
        return static_cast<bool>(self.count(p));
      });

  py::enum_<WireBundle>(m, "WireBundle", py::arithmetic())
      .value("Core", WireBundle::Core)
      .value("DMA", WireBundle::DMA)
      .value("FIFO", WireBundle::FIFO)
      .value("South", WireBundle::South)
      .value("West", WireBundle::West)
      .value("North", WireBundle::North)
      .value("East", WireBundle::East)
      .value("PLIO", WireBundle::PLIO)
      .value("NOC", WireBundle::NOC)
      .value("Trace", WireBundle::Trace)
      .export_values();

  py::class_<Port>(m, "Port")
      .def(py::init<Port &>())
      .def(py::init<WireBundle, int>())
      .def_readonly("bundle", &Port::bundle)
      .def_readwrite("channel", &Port::channel)
      .def(py::hash(py::self))
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def("__repr__", [](const Port &port) { return to_string(port); });

  py::class_<TileID>(m, "TileID")
      .def(py::init<TileID &>())
      .def_readonly("col", &TileID::col)
      .def_readonly("row", &TileID::row)
      // Implements __hash__ (magic?)
      .def(py::hash(py::self))
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def("__repr__", [](const TileID &tile) { return to_string(tile); });

  py::class_<Switchbox, TileID>(m, "Switchbox")
      .def(py::init<Switchbox &>())
      .def(py::init<int, int>())
      // Implements __hash__ (magic?)
      .def(py::hash(py::self))
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def("__repr__", [](const Switchbox &sb) { return to_string(sb); });

  py::class_<SwitchSetting>(m, "SwitchSetting")
      .def(py::init<SwitchSetting &>())
      .def(py::init<Port &>())
      .def_readonly("src", &SwitchSetting::src)
      .def_readwrite("dsts", &SwitchSetting::dsts)
      .def(py::self < py::self)
      .def("__repr__", [](const SwitchSetting &sb) { return to_string(sb); });

  py::class_<Channel>(m, "Channel")
      .def(py::init<Channel &>())
      .def_property_readonly("src", [](const Channel &c) { return c.src; })
      .def_property_readonly("target",
                             [](const Channel &c) { return c.target; })
      .def_readonly("bundle", &Channel::bundle)
      .def_readonly("demand", &Channel::demand)
      // Probably don't need these...
      .def("__repr__", [](const Channel &c) { return to_string(c); });

  py::class_<PathEndPoint>(m, "PathEndPoint")
      .def(py::init<PathEndPoint &>())
      .def_readonly("sb", &PathEndPoint::sb)
      .def_readonly("port", &PathEndPoint::port)
      // Implements __hash__ (magic?)
      .def(py::hash(py::self))
      .def(py::self == py::self)
      .def(py::self < py::self)
      .def("__repr__", [](const PathEndPoint &pe) { return to_string(pe); });

  py::class_<Flow>(m, "Flow")
      .def(py::init<Flow &>())
      .def_readonly("src", &Flow::src)
      .def_readonly("dsts", &Flow::dsts)
      .def("__repr__", [](const Flow &flow) { return to_string(flow); });

  py::class_<AIETargetModel>(m, "AIETargetModel")
      .def("get_num_source_switchbox_connections",
           &AIETargetModel::getNumSourceSwitchboxConnections,
           // Something about the factor that AIETargetModel is virtual
           // (and thus always a pointer) necessitates this.
           py::return_value_policy::reference)
      .def("get_num_dest_switchbox_connections",
           &AIETargetModel::getNumDestSwitchboxConnections,
           py::return_value_policy::reference);
}
