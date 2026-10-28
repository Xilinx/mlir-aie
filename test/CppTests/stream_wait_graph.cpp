//===- stream_wait_graph.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace xilinx::AIE;

int main() {
  MLIRContext context;
  context.getOrLoadDialect<AIEDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
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

  // On memtile (0, 1), S and T stream into receiver R. S's data reaches R
  // first, taking, where T is no wait of R's. R reached again free, through
  // the lock waits S -> C1 -> C2 -> B -> R, waits on T. R also waits on B.
  auto chainModule = parseSourceString<ModuleOp>(R"mlir(
    module {
      aie.device(npu1) {
        %m = aie.tile(0, 1)
        %buf = aie.buffer(%m) : memref<16xi32>
        %la = aie.lock(%m, 0) {init = 0 : i32}
        %lb = aie.lock(%m, 1) {init = 0 : i32}
        %lc = aie.lock(%m, 2) {init = 0 : i32}
        %ld = aie.lock(%m, 3) {init = 0 : i32}
        %lt = aie.lock(%m, 4) {init = 1 : i32}
        %lz = aie.lock(%m, 5) {init = 0 : i32}
        %dma = aie.memtile_dma(%m) {
          %one = arith.constant 1 : i32
          %0 = aie.dma_start(MM2S, 0, ^s, ^c1_start)
        ^s:
          aie.use_lock(%la, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%lz, Release, %one)
          aie.next_bd ^s
        ^c1_start:
          %1 = aie.dma_start(MM2S, 1, ^c1, ^c2_start)
        ^c1:
          aie.use_lock(%lb, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%la, Release, %one)
          aie.next_bd ^c1
        ^c2_start:
          %2 = aie.dma_start(MM2S, 2, ^c2, ^b_start)
        ^c2:
          aie.use_lock(%lc, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%lb, Release, %one)
          aie.next_bd ^c2
        ^b_start:
          %3 = aie.dma_start(MM2S, 3, ^b, ^t_start)
        ^b:
          aie.use_lock(%ld, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%lc, Release, %one)
          aie.next_bd ^b
        ^t_start:
          %4 = aie.dma_start(MM2S, 4, ^t, ^r_start)
        ^t:
          aie.use_lock(%lt, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%lt, Release, %one)
          aie.next_bd ^t
        ^r_start:
          %5 = aie.dma_start(S2MM, 0, ^r, ^end)
        ^r:
          aie.use_lock(%lc, AcquireGreaterEqual, %one)
          aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
          aie.use_lock(%ld, Release, %one)
          aie.next_bd ^r
        ^end:
          aie.end
        }
      }
    })mlir",
                                                 &context);
  if (!chainModule)
    return 1;
  auto chainDevice = *chainModule->getOps<DeviceOp>().begin();
  auto channel = [](int ch) {
    return StreamEndpoint{{0, 1}, {WireBundle::DMA, ch}};
  };
  SmallVector<RoutedStream, 2> streams{{channel(0), channel(0), {}},
                                       {channel(4), channel(0), {}}};
  StreamVolumeAnalysis volumes(chainDevice, streams);
  StreamWaitGraph graph(chainDevice, streams, volumes);
  auto mm2s = [&](int ch) { return graph.agentAt(channel(ch), true); };
  auto s = mm2s(0), c1 = mm2s(1), c2 = mm2s(2), b = mm2s(3), t = mm2s(4);
  auto r = graph.agentAt(channel(0), false);
  if (!s || !c1 || !c2 || !b || !t || !r)
    return 1;
  if (graph.waitChain({*s}, {*t}, {}) !=
          SmallVector<unsigned>({*s, *c1, *c2, *b, *r, *t}) ||
      !graph.waitChain({*s}, {*t}, {*c1}).empty()) {
    llvm::errs() << "Incorrect wait chain through a receiver reached again\n";
    return 1;
  }
  return 0;
}
