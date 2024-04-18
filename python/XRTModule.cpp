//===- XRTModule.cpp -------------------------------------------000---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

// group_id 0 is for ipu instructions
// group_id 1 is for number of ipu instructions
// host side buffers/args follow starting from position 2
// see aiecc.main.emit_design_kernel_json
constexpr size_t HOST_BUFFERS_START_IDX = 2;

class PyXCLBin {
public:
  PyXCLBin(const std::string &xclBinPath, const std::string &kernelName,
           int deviceIndex)
      : xclBin(std::make_unique<xrt::xclbin>(xclBinPath)),
        device(std::make_unique<xrt::device>(deviceIndex)) {
    assert(device->get_info<xrt::info::device::name>() == "RyzenAI-Phoenix" &&
           "only Phoenix supported by xrt python bindings");
    device->register_xclbin(*xclBin);
    context = std::make_unique<xrt::hw_context>(*device, xclBin->get_uuid());
    kernel = std::make_unique<xrt::kernel>(*context, kernelName);
  }

  void loadNPUInstructions(const std::vector<uint32_t> &insts) {
    ipuInstructions =
        std::make_unique<xrt::bo>(*device, insts.size() * sizeof(uint32_t),
                                  XCL_BO_FLAGS_CACHEABLE, kernel->group_id(0));
    uint32_t *bufInstr = ipuInstructions->map<uint32_t *>();
    for (size_t i = 0; i < insts.size(); ++i)
      bufInstr[i] = insts.at(i);
    ipuInstructions->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  template <typename ElementT>
  std::vector<py::memoryview>
  mmapBuffers(std::vector<std::vector<int>> shapes) {
    this->buffers.reserve(shapes.size());
    std::vector<py::memoryview> views;
    views.reserve(shapes.size());

    auto initAndViewBuffer = [this](
                                 std::vector<int> shape, int groupId,
                                 std::vector<std::unique_ptr<xrt::bo>> &buffers,
                                 std::vector<py::memoryview> &views) {
      int nElements =
          std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
      int nBytes = nElements * sizeof(ElementT);
      xrt::bo xrtBuf(*device, nBytes, XRT_BO_FLAGS_HOST_ONLY,
                     kernel->group_id(groupId));
      buffers.push_back(std::make_unique<xrt::bo>(xrtBuf));

      ElementT *buf = xrtBuf.map<ElementT *>();
      for (int i = 0; i < nElements; ++i)
        buf[i] = static_cast<ElementT>(0);

      std::vector strides_{1};
      for (int i = shape.size() - 1; i > 0; i--)
        strides_.push_back(strides_.back() * shape[i]);
      std::vector<int> strides;
      // stride in bytes
      std::transform(strides_.rbegin(), strides_.rend(),
                     std::back_inserter(strides),
                     [](int s) { return s * sizeof(ElementT); });
      views.push_back(py::memoryview::from_buffer(buf, shape, strides));
    };

    for (size_t i = 0; i < shapes.size(); ++i)
      initAndViewBuffer(shapes[i], HOST_BUFFERS_START_IDX + i, this->buffers,
                        views);
    return views;
  }

  uint64_t getBufferHostAddress(size_t idx) { return buffers[idx]->address(); }

  void syncBuffersToDevice() {
    for (auto &buf : this->buffers)
      buf->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void syncBuffersFromDevice() {
    for (auto &buf : this->buffers)
      buf->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  void run() {
    run_ = std::make_unique<xrt::run>(*kernel);
    run_->set_arg(0, *ipuInstructions);
    run_->set_arg(1, ipuInstructions->size());
    for (size_t i = 0; i < buffers.size(); ++i)
      run_->set_arg(HOST_BUFFERS_START_IDX + i, *buffers[i]);
    run_->start();
  }

  void _runOnlyNpuInstructions() {
    run_ = std::make_unique<xrt::run>(*kernel);
    run_->set_arg(0, *ipuInstructions);
    run_->set_arg(1, ipuInstructions->size());
    run_->start();
  }

  void wait(const std::optional<int> timeout) {
    if (timeout) {
      if (run_->wait(timeout.value() * 1000) == ERT_CMD_STATE_TIMEOUT)
        throw std::runtime_error("kernel timed out");
    } else
      (void)run_->wait();
  }

  std::unique_ptr<xrt::xclbin> xclBin;
  std::unique_ptr<xrt::device> device;
  std::unique_ptr<xrt::hw_context> context;
  std::unique_ptr<xrt::kernel> kernel;
  std::unique_ptr<xrt::bo> ipuInstructions;

  std::vector<std::unique_ptr<xrt::bo>> buffers;

  std::unique_ptr<xrt::run> run_;
};

PYBIND11_MODULE(_xrt, m) {

  py::class_<PyXCLBin>(m, "XCLBin", py::module_local())
      .def(py::init<const std::string &, const std::string &, int>(),
           "xclbin_path"_a, "kernel_name"_a, "device_index"_a = 0)
      .def("load_ipu_instructions", &PyXCLBin::loadNPUInstructions, "insts"_a)
      .def("sync_buffers_to_device", &PyXCLBin::syncBuffersToDevice)
      .def("sync_buffers_from_device", &PyXCLBin::syncBuffersFromDevice)
      .def("run", &PyXCLBin::run)
      .def("_run_only_ipu_instructions", &PyXCLBin::_runOnlyNpuInstructions)
      .def("wait", &PyXCLBin::wait, "timeout"_a = py::none())
      .def(
          "mmap_buffers",
          [](PyXCLBin &self, const std::vector<std::vector<int>> &shapes,
             const py::object &npFormat) {
            auto npy = py::module_::import("numpy");
            if (npFormat.is(npy.attr("int16")))
              return self.mmapBuffers<int16_t>(shapes);
            if (npFormat.is(npy.attr("int32")))
              return self.mmapBuffers<int32_t>(shapes);
            if (npFormat.is(npy.attr("float32")))
              return self.mmapBuffers<float>(shapes);
            if (npFormat.is(npy.attr("int64")))
              return self.mmapBuffers<int64_t>(shapes);
            if (npFormat.is(npy.attr("float64")))
              return self.mmapBuffers<double>(shapes);
            throw std::runtime_error("unsupported np format: " +
                                     py::repr(npFormat).cast<std::string>());
          },
          "shapes"_a, "np_format"_a)
      .def("_get_buffer_host_address", [](PyXCLBin &self, size_t idx) {
        return self.getBufferHostAddress(idx);
      });
  ;
}
