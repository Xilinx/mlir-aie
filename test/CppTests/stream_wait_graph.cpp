//===- stream_wait_graph.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace xilinx::AIE;

int main() {
  MLIRContext context;
  context.getOrLoadDialect<AIEDialect>();
  auto module =
      parseSourceString<ModuleOp>("module { aie.device(npu1) {} }", &context);
  if (!module)
    return 1;
  auto device = *module->getOps<DeviceOp>().begin();

  for (WireBundle receiver : {WireBundle::Core, WireBundle::DMA}) {
    for (WireBundle sender : {WireBundle::Core, WireBundle::DMA}) {
      SmallVector<RoutedStream, 2> streams{
          {{{0, 2}, {WireBundle::DMA, 0}}, {{0, 3}, {receiver, 0}}, {}},
          {{{0, 4}, {sender, 0}},
           {{0, 3}, {receiver, receiver == WireBundle::Core ? 1 : 0}},
           {}}};
      StreamVolumeAnalysis volumes(device, streams);
      StreamWaitGraph graph(device, streams, volumes);
      auto s = graph.agentAt(streams[0].src, true);
      auto r = graph.agentAt(streams[0].dst, false);
      auto t = graph.agentAt(streams[1].src, true);
      if (!s || !r || !t)
        return 1;

      SmallVector<unsigned> expected;
      if (receiver != WireBundle::DMA || sender != WireBundle::DMA)
        expected = {*s, *r, *t};
      if (graph.waitChain({*s}, {*t}, {}) != expected ||
          !graph.waitChain({*s}, {*t}, {*r}).empty() ||
          graph.waitChain({*r}, {*t}, {}) != SmallVector<unsigned>({*r, *t})) {
        llvm::errs() << "Incorrect wait chain for receiver "
                     << stringifyWireBundle(receiver) << " and sender "
                     << stringifyWireBundle(sender) << "\n";
        return 1;
      }
    }
  }
  return 0;
}
