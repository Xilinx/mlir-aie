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

using namespace mlir;
using namespace mlir::python::adaptors;
using namespace xilinx::AIE;

namespace py = pybind11;

void xilinx::AIE::bindTypes(py::module_ &m) {

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
      .def_readonly("bundle", &Port::bundle)
      .def_readonly("channel", &Port::channel)
      .def("__str__", [](const Port &port) { return to_string(port); })
      .def("__repr__", [](const Port &port) { return to_string(port); });

  py::class_<TileID>(m, "TileID")
      .def(py::init<TileID &>())
      .def_readonly("col", &TileID::col)
      .def_readonly("row", &TileID::row)
      .def("__str__", [](const TileID &tile) { return to_string(tile); })
      .def("__repr__", [](const TileID &tile) { return to_string(tile); });

  py::class_<Switchbox, TileID>(m, "Switchbox")
      .def(py::init<Switchbox &>())
      .def("__str__", [](const Switchbox &sb) { return to_string(sb); })
      .def("__repr__", [](const Switchbox &sb) { return to_string(sb); });

  py::class_<Channel>(m, "Channel")
      .def(py::init<Channel &>())
      .def_property_readonly("src", [](const Channel &c) { return c.src; })
      .def_property_readonly("target",
                             [](const Channel &c) { return c.target; })
      .def_readonly("bundle", &Channel::bundle)
      .def_readonly("max_capacity", &Channel::maxCapacity)
      .def_readonly("demand", &Channel::demand)
      // Probably don't need these...
      .def_readonly("used_capacity", &Channel::usedCapacity)
      .def_readonly("fixed_capacity", &Channel::fixedCapacity)
      .def_readonly("over_capacity_count", &Channel::overCapacityCount)
      .def("__str__", [](const Channel &c) { return to_string(c); })
      .def("__repr__", [](const Channel &c) { return to_string(c); });

  py::class_<PathEndPoint>(m, "PathEndPoint")
      .def(py::init<PathEndPoint &>())
      .def_readonly("sb", &PathEndPoint::sb)
      .def_readonly("port", &PathEndPoint::port)
      .def("__str__", [](const PathEndPoint &pe) { return to_string(pe); })
      .def("__repr__", [](const PathEndPoint &pe) { return to_string(pe); });

  py::class_<Flow>(m, "Flow")
      .def(py::init<Flow &>())
      .def_readonly("src", &Flow::src)
      .def_readonly("dsts", &Flow::dsts)
      .def("__str__", [](const Flow &flow) { return to_string(flow); })
      .def("__repr__", [](const Flow &flow) { return to_string(flow); });
}