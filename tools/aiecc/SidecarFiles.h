//===- SidecarFiles.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Factories for the non-MLIR text/JSON sidecar files that aiecc hands to
// external tools (bootgen, xclbinutil, aiebu-asm). Each function returns the
// payload value; serialization to disk happens through Item<T>::asFile().
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_SIDECARFILES_H
#define AIECC_SIDECARFILES_H

#include "Graph.h"
#include "IRTransforms.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <optional>
#include <random>
#include <string>

namespace xilinx::aiecc {

// Key that uniquely identifies one runtime sequence as "<device>_<sequence>".
// Shared by the per-runtime-sequence NPU-insts edge and the full-ELF config so
// both sides agree on how a sequence's `.bin` is keyed.
inline std::string npuSeqKey(llvm::StringRef deviceName,
                             llvm::StringRef sequenceName) {
  return deviceName.str() + "_" + sequenceName.str();
}

// BIF text consumed by bootgen to assemble a device's PDI from its CDOs.
inline std::string makeBifText(llvm::StringRef cdoDir,
                               llvm::StringRef devName) {
  return llvm::formatv(R"(all:
{{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  image
  {{
    name=aie_image, id=0x1c000000
    {{ type=cdo
      file={0}/{1}_aie_cdo_elfs.bin
      file={0}/{1}_aie_cdo_init.bin
      file={0}/{1}_aie_cdo_enable.bin
    }
  }
}
)",
                       cdoDir, devName)
      .str();
}

// MEM_TOPOLOGY JSON — fixed (HOST + SRAM DRAM regions).
inline llvm::json::Value makeMemTopologyJson() {
  auto dram = [](llvm::StringRef sizeKB, llvm::StringRef tag) {
    return llvm::json::Object{{"m_type", "MEM_DRAM"},
                              {"m_used", "1"},
                              {"m_sizeKB", sizeKB},
                              {"m_tag", tag},
                              {"m_base_address", "0x4000000"}};
  };
  return llvm::json::Object{
      {"mem_topology",
       llvm::json::Object{
           {"m_count", "2"},
           {"m_mem_data", llvm::json::Array{dram("0x10000", "HOST"),
                                            dram("0xc000", "SRAM")}}}}};
}

//===----------------------------------------------------------------------===//
// Host-buffer (boN) count
//===----------------------------------------------------------------------===//

// The NPU firmware command-chain (xrt::runlist) ABI requires every kernel to
// declare at least this many host buffer slots; fewer produces an undersized
// command slot and the runlist aborts. Extra slots are harmless.
inline constexpr int kMinHostBOs = 5;

// Conservative, hardware-verified ceiling on host buffer slots. AIETargetNPU
// folds the DDR translation offset so >5 buffers work, but counts above this
// are unvalidated and rejected.
inline constexpr int kMaxHostBOs = 16;

// Argument count of `dev`'s selected non-empty runtime sequence, or nullopt.
// A non-empty `seqFilter` (--sequence-name) restricts the choice to that
// sequence, matching the instruction/control-packet paths.
inline std::optional<int>
getRuntimeSequenceArgCount(xilinx::AIE::DeviceOp dev,
                           llvm::StringRef seqFilter) {
  for (auto seqOp : dev.getOps<xilinx::AIE::RuntimeSequenceOp>()) {
    if (!seqFilter.empty() && seqOp.getSymName() != seqFilter)
      continue;
    if (!seqOp.getBody().empty())
      return static_cast<int>(seqOp.getBody().front().getNumArguments());
  }
  return std::nullopt;
}

// True if `dev` carries a control-packet shim DMA allocation (symbol prefixed
// "ctrlpkt"), i.e. the host passes one extra control-packet buffer.
inline bool deviceUsesControlPackets(xilinx::AIE::DeviceOp dev) {
  for (auto allocOp : dev.getOps<xilinx::AIE::ShimDMAAllocationOp>())
    if (allocOp.getSymName().starts_with("ctrlpkt"))
      return true;
  return false;
}

// Number of host buffer (boN) args kernels.json must declare for `target`: the
// runtime sequence arity, floored at kMinHostBOs. A device with no sequence of
// its own (the ctrl-packet reconfig "base" xclbin) derives its count from the
// unique sibling device that has one, plus one control-packet buffer when
// `target` carries a ctrlpkt shim allocation; failing that it falls back to the
// floor.
inline int computeNumHostBOs(xilinx::AIE::DeviceOp target,
                             llvm::StringRef seqFilter) {
  if (auto own = getRuntimeSequenceArgCount(target, seqFilter))
    return std::max(kMinHostBOs, *own);

  std::optional<int> sibling;
  bool unique = true;
  if (auto mod = target->getParentOfType<mlir::ModuleOp>())
    for (auto dev : mod.getOps<xilinx::AIE::DeviceOp>()) {
      if (dev == target)
        continue;
      if (auto c = getRuntimeSequenceArgCount(dev, seqFilter)) {
        if (sibling) {
          unique = false;
          break;
        }
        sibling = c;
      }
    }
  if (sibling && unique)
    return std::max(kMinHostBOs,
                    *sibling + (deviceUsesControlPackets(target) ? 1 : 0));
  return kMinHostBOs;
}

inline llvm::json::Value makeKernelsJson(llvm::StringRef kernelName,
                                         llvm::StringRef instanceName,
                                         llvm::StringRef kernelId,
                                         int numHostBOs) {
  using O = llvm::json::Object;
  auto hex = [](int v) {
    return llvm::formatv("0x{0}", llvm::utohexstr(v)).str();
  };
  llvm::json::Array arguments{O{{"name", "opcode"},
                                {"address-qualifier", "SCALAR"},
                                {"type", "uint64_t"},
                                {"offset", "0x00"}},
                              O{{"name", "instr"},
                                {"memory-connection", "SRAM"},
                                {"address-qualifier", "GLOBAL"},
                                {"type", "char *"},
                                {"offset", "0x08"}},
                              O{{"name", "ninstr"},
                                {"address-qualifier", "SCALAR"},
                                {"type", "uint32_t"},
                                {"offset", "0x10"}}};
  for (int i = 0, off = 0x14; i < numHostBOs; ++i, off += 8)
    arguments.push_back(O{{"name", ("bo" + llvm::Twine(i)).str()},
                          {"memory-connection", "HOST"},
                          {"address-qualifier", "GLOBAL"},
                          {"type", "void*"},
                          {"offset", hex(off)}});
  return O{
      {"ps-kernels",
       O{{"kernels",
          llvm::json::Array{O{
              {"name", kernelName},
              {"type", "dpu"},
              {"extended-data", O{{"subtype", "DPU"},
                                  {"functional", "0"},
                                  {"dpu_kernel_id", kernelId}}},
              {"arguments", std::move(arguments)},
              {"instances", llvm::json::Array{O{{"name", instanceName}}}}}}}}}};
}

// v4 UUID used as the PDI uuid in the partition JSON.
inline std::string generatePdiUUID() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
  uint32_t data[4];
  for (int i = 0; i < 4; ++i)
    data[i] = dis(gen);
  data[1] = (data[1] & 0xFFFF0FFF) | 0x4000;
  data[2] = (data[2] & 0x3FFFFFFF) | 0x80000000;
  return llvm::formatv("{0:x-8}-{1:x-4}-{2:x-4}-{3:x-4}-{4:x-12}", data[0],
                       data[1] >> 16, data[1] & 0xFFFF, data[2] >> 16,
                       ((uint64_t)(data[2] & 0xFFFF) << 32) | data[3]);
}

inline llvm::json::Value makePartitionJson(xilinx::AIE::DeviceOp devOp,
                                           llvm::StringRef pdiPath,
                                           llvm::StringRef kernelId) {
  using O = llvm::json::Object;
  int numCols = devOp.getTargetModel().columns();
  auto device = devOp.getDevice();
  llvm::json::Array startColumns;
  if (device == xilinx::AIE::AIEDevice::npu1 ||
      device == xilinx::AIE::AIEDevice::npu2)
    startColumns.push_back(0);
  else
    for (int i = 1; i < 6 - numCols; ++i)
      startColumns.push_back(i);
  return O{
      {"aie_partition",
       O{{"name", "QoS"},
         {"operations_per_cycle", "2048"},
         {"inference_fingerprint", "23423"},
         {"pre_post_fingerprint", "12345"},
         {"partition", O{{"column_width", numCols},
                         {"start_columns", std::move(startColumns)}}},
         {"PDIs", llvm::json::Array{
                      O{{"uuid", generatePdiUUID()},
                        {"file_name", pdiPath.str()},
                        {"cdo_groups",
                         llvm::json::Array{O{
                             {"name", "DPU"},
                             {"type", "PRIMARY"},
                             {"pdi_id", "0x01"},
                             {"dpu_kernel_ids", llvm::json::Array{kernelId}},
                             {"pre_cdo_groups",
                              llvm::json::Array{std::string("0xC1")}}}}}}}}}}};
}

// Patch-info JSON for external buffers (the runtime control-packet buffer).
// Consumed by `aiebu-asm` via the `patch_info_file` field in the full-ELF
// config JSON. `ctrlPktArgIdx` is the runtime-sequence argument slot the
// control-packet buffer occupies (the sequence's pre-lowering argument count,
// since ctrl-packet-to-DMA appends the ctrl buffer as the next argument).
inline llvm::json::Value makePatchInfoJson(int ctrlPktArgIdx,
                                           int64_t ctrlPktSizeBytes) {
  using O = llvm::json::Object;
  return O{{"external_buffers",
           O{{"buffer_ctrl",
              O{{"xrt_id", ctrlPktArgIdx},
                {"logical_id", -1},
                {"size_in_bytes", ctrlPktSizeBytes},
                {"ctrl_pkt_buffer", 1},
                {"name", "runtime_control_packet"}}}}}};
}

// Full-ELF config.json fed to `aiebu-asm -t aie2_config`. One xrt-kernel per
// device with ≥1 runtime sequence; PDIs array is shared (all devices) so
// aiebu-asm can resolve any load_pdi reference. Argument count is
// max(3, max runtime-seq arity). PDI IDs are read from `aiecc.pdi_id` on each
// DeviceOp (stamped by `assignDevicePdiIds`).
//
// `ctrlPktPaths` / `patchInfoPaths` (both device-keyed, optional) carry the
// per-device control-packet binary and its patch-info JSON for the
// load-pdi-to-ctrl-pkt reconfigure flow; when present they are attached to the
// device's runtime-sequence instances.
inline llvm::json::Value makeFullElfConfigJson(
    const Node<OpInModule<xilinx::AIE::DeviceOp>> &devices,
    const llvm::StringMap<std::string> &pdiPaths,
    const llvm::StringMap<std::string> &instsPaths,
    const llvm::StringMap<std::string> &ctrlPktPaths = {},
    const llvm::StringMap<std::string> &patchInfoPaths = {}) {
  using O = llvm::json::Object;
  auto devId = [](xilinx::AIE::DeviceOp d) {
    return static_cast<int>(
        d->getAttrOfType<mlir::IntegerAttr>(kPdiIdAttr).getInt());
  };

  llvm::json::Array allPdis;
  for (const auto &item : devices.items)
    if (auto it = pdiPaths.find(item.key); it != pdiPaths.end())
      allPdis.push_back(
          O{{"id", devId(item.get().op)}, {"PDI_file", it->second}});

  llvm::json::Array xrtKernels;
  for (const auto &item : devices.items) {
    const std::string &devName = item.key;
    xilinx::AIE::DeviceOp devOp = item.get().op;

    int argCount = 3;
    devOp.walk([&](xilinx::AIE::RuntimeSequenceOp seq) {
      if (!seq.getBody().empty())
        argCount =
            std::max<int>(argCount, seq.getBody().front().getNumArguments());
    });
    llvm::json::Array arguments;
    for (int i = 0; i < argCount; ++i)
      arguments.push_back(
          O{{"name", "arg_" + std::to_string(i)},
            {"type", "char *"},
            {"offset", llvm::formatv("0x{0}", llvm::utohexstr(i * 8)).str()}});

    auto ctrlPktIt = ctrlPktPaths.find(devName);
    auto patchIt = patchInfoPaths.find(devName);

    llvm::json::Array instances;
    devOp.walk([&](xilinx::AIE::RuntimeSequenceOp seq) {
      // One `.bin` per runtime sequence, keyed "<device>_<sequence>". Only
      // sequences that were actually lowered to a control-code binary have an
      // entry in `instsPaths`; skip the rest (e.g. filtered out via
      // `--sequence-name`) so we don't emit an instance with an empty
      // TXN_ctrl_code_file, which is invalid for `aiebu-asm -t aie2_config`.
      auto instsIt = instsPaths.find(npuSeqKey(devName, seq.getSymName()));
      if (instsIt == instsPaths.end())
        return;
      O inst{{"id", seq.getSymName().str()},
             {"TXN_ctrl_code_file", instsIt->second}};
      if (ctrlPktIt != ctrlPktPaths.end())
        inst["ctrl_packet_file"] = ctrlPktIt->second;
      if (patchIt != patchInfoPaths.end())
        inst["patch_info_file"] = patchIt->second;
      instances.push_back(std::move(inst));
    });
    if (instances.empty())
      continue;

    llvm::json::Array pdisCopy;
    for (const auto &p : allPdis)
      pdisCopy.push_back(llvm::json::Value(p));

    xrtKernels.push_back(O{{"name", devName},
                           {"arguments", std::move(arguments)},
                           {"PDIs", std::move(pdisCopy)},
                           {"instance", std::move(instances)}});
  }

  return O{{"xrt-kernels", std::move(xrtKernels)}};
}

// external_buffers.json patch fed to `aiebu-asm -t aie2txn -j` (or the
// in-memory `aiebu_assembler_get_elf` patch argument) when assembling a
// combined control-packet ELF. Describes the runtime control-packet buffer:
// `xrt_id` is the control buffer's argument slot (the runtime sequence's
// pre-lowering argument count, since ctrl-packet-to-DMA appends the ctrl
// buffer as the next argument) and `size_in_bytes` is the control-packet
// binary size.
inline llvm::json::Value makeCtrlpktExtBufJson(xilinx::AIE::DeviceOp devOp,
                                               uint64_t ctrlPktSize,
                                               llvm::StringRef seqFilter) {
  using O = llvm::json::Object;
  int ctrlIdx = getRuntimeSequenceArgCount(devOp, seqFilter).value_or(0);
  return O{
      {"external_buffers",
       O{{"buffer_ctrl", O{{"xrt_id", ctrlIdx},
                           {"logical_id", -1},
                           {"size_in_bytes", static_cast<int64_t>(ctrlPktSize)},
                           {"ctrl_pkt_buffer", 1},
                           {"name", "runtime_control_packet"}}}}}};
}

} // namespace xilinx::aiecc

#endif // AIECC_SIDECARFILES_H
