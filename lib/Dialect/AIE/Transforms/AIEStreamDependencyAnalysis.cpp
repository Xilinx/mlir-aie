//===- AIEStreamDependencyAnalysis.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIERoutingDiagnostics.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <deque>
#include <set>
#include <variant>

#define DEBUG_TYPE "aie-stream-dependency"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

std::optional<TileID> tileOf(Value tile) {
  if (auto tileOp = dyn_cast_or_null<TileOp>(tile.getDefiningOp()))
    return tileOp.getTileID();
  return std::nullopt;
}

std::optional<DmaChannelProgram> makeProgram(Operation *op, DeviceOp device) {
  auto parentTile = [](Operation *op) -> std::optional<TileID> {
    if (auto element = op->getParentOfType<TileElement>())
      return tileOf(element.getTile());
    return std::nullopt;
  };
  if (auto start = dyn_cast<DMAStartOp>(op)) {
    std::optional<TileID> tile = parentTile(op);
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{op,
                        {*tile, start.getChannelDir(),
                         static_cast<int>(start.getChannelIndex())}};
    p.passes += start.getRepeatCount();
    llvm::SmallPtrSet<Block *, 8> seen;
    for (Block *b = start.getDest(); b;) {
      if (!seen.insert(b).second) {
        p.loops = true;
        p.loopStart = llvm::find(p.bds, b) - p.bds.begin();
        break;
      }
      p.bds.push_back(b);
      auto next = dyn_cast<NextBDOp>(b->getTerminator());
      b = next ? next.getDest() : nullptr;
    }
    return p;
  }
  if (auto dma = dyn_cast<DMAOp>(op)) {
    std::optional<TileID> tile = parentTile(op);
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{
        op,
        {*tile, dma.getChannelDir(), static_cast<int>(dma.getChannelIndex())}};
    p.passes += dma.getRepeatCount();
    for (Region &r : dma.getBds())
      p.regions.push_back(&r);
    p.loops = dma.getLoop();
    return p;
  }
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
    std::optional<TileID> tile = tileOf(task.getTile());
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{
        op, {*tile, task.getDirection(), static_cast<int>(task.getChannel())}};
    p.regions.push_back(&task.getBody());
    p.passes += task.getRepeatCount();
    return p;
  }
  if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
    auto alloc = ShimDMAAllocationOp::getForSymbol(
        device, task.getAlloc().getRootReference());
    if (!alloc)
      return std::nullopt;
    std::optional<TileID> tile = tileOf(alloc.getTile());
    if (!tile)
      return std::nullopt;
    DmaChannelProgram p{op,
                        {*tile, alloc.getChannelDir(),
                         static_cast<int>(alloc.getChannelIndex())}};
    p.regions.push_back(&task.getBody());
    p.passes += task.getRepeatCount();
    return p;
  }
  return std::nullopt;
}

// Every DMA channel program: aie.dma_start and aie.dma inside the tile DMA
// ops, and runtime-configured tasks.
SmallVector<DmaChannelProgram> collectDmaPrograms(DeviceOp device) {
  SmallVector<DmaChannelProgram> programs;
  device.walk([&](Operation *op) {
    if (isa<DMAStartOp, DMAOp, AIEX::DMAConfigureTaskOp,
            AIEX::DMAConfigureTaskForOp>(op))
      if (std::optional<DmaChannelProgram> p = makeProgram(op, device))
        programs.push_back(std::move(*p));
  });
  return programs;
}

template <typename OpT>
void forEachInProgram(const DmaChannelProgram &p,
                      llvm::function_ref<void(OpT)> fn) {
  for (Block *b : p.bds)
    for (auto op : b->getOps<OpT>())
      fn(op);
  for (Region *r : p.regions)
    r->walk([&](OpT op) { fn(op); });
}

std::optional<TileDMAChannel> channelOfSymbol(DeviceOp device,
                                              StringRef symbol) {
  auto alloc = ShimDMAAllocationOp::getForSymbol(device, symbol);
  if (!alloc)
    return std::nullopt;
  std::optional<TileID> tile = tileOf(alloc.getTile());
  if (!tile)
    return std::nullopt;
  return TileDMAChannel{*tile, alloc.getChannelDir(),
                        static_cast<int>(alloc.getChannelIndex())};
}

// The ops that can create `task`, following it back through the results and
// block arguments of runtime control flow (scf.for iter_args, scf.if results).
// Empty when some value it can come from has no creator here.
SmallVector<Operation *> taskCreators(Value task) {
  SmallVector<Operation *> creators;
  llvm::SmallPtrSet<Value, 8> seen;
  SmallVector<Value> worklist{task};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *op = v.getDefiningOp();
    if (isa_and_nonnull<AIEX::DMAConfigureTaskOp, AIEX::DMAConfigureTaskForOp,
                        AIEX::DMAStartBdChainOp, AIEX::DMAStartBdChainForOp>(
            op)) {
      creators.push_back(op);
      continue;
    }
    Operation *owner =
        op ? op : cast<BlockArgument>(v).getOwner()->getParentOp();
    auto branch = dyn_cast_or_null<RegionBranchOpInterface>(owner);
    if (!branch)
      return {};
    RegionBranchInverseSuccessorMapping mapping;
    branch.getSuccessorInputOperandMapping(mapping);
    auto it = mapping.find(v);
    if (it == mapping.end())
      return {};
    for (OpOperand *operand : it->second)
      worklist.push_back(operand->get());
  }
  return creators;
}

// The channels `task` can run on; empty when that is not known here.
SmallVector<TileDMAChannel> channelsOfTask(DeviceOp device, Value task) {
  SmallVector<TileDMAChannel> keys;
  for (Operation *op : taskCreators(task)) {
    std::optional<TileDMAChannel> key;
    auto onTile = [&](Value tileValue, DMAChannelDir dir, int channel) {
      if (std::optional<TileID> tile = tileOf(tileValue))
        key = TileDMAChannel{*tile, dir, channel};
    };
    if (auto configure = dyn_cast<AIEX::DMAConfigureTaskOp>(op))
      onTile(configure.getTile(), configure.getDirection(),
             configure.getChannel());
    else if (auto chain = dyn_cast<AIEX::DMAStartBdChainOp>(op))
      onTile(chain.getTile(), chain.getDirection(), chain.getChannel());
    else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op))
      key = channelOfSymbol(device, configure.getAlloc().getRootReference());
    else if (auto chain = dyn_cast<AIEX::DMAStartBdChainForOp>(op))
      key = channelOfSymbol(device, chain.getAlloc());
    if (!key)
      return {};
    if (!llvm::is_contained(keys, *key))
      keys.push_back(*key);
  }
  return keys;
}

// Visits the ops in `region` in program order, going through the body of
// each loop twice, so that what one iteration does follows all the previous
// iteration did.
void walkLoopsTwice(Region &region, function_ref<void(Operation *)> fn) {
  for (Operation &op : region.getOps()) {
    fn(&op);
    auto branch = dyn_cast<RegionBranchOpInterface>(op);
    int passes = branch && branch.hasLoop() ? 2 : 1;
    for (int pass = 0; pass < passes; pass++)
      for (Region &r : op.getRegions())
        walkLoopsTwice(r, fn);
  }
}

// Packet ids each sending channel is programmed with.
std::map<TileDMAChannel, std::set<int>> collectSentPacketIDs(DeviceOp device) {
  std::map<TileDMAChannel, std::set<int>> ids;
  for (const DmaChannelProgram &p : collectDmaPrograms(device)) {
    if (p.dma.dir != DMAChannelDir::MM2S)
      continue;
    forEachInProgram<DMABDOp>(p, [&](DMABDOp bd) {
      if (std::optional<PacketInfoAttr> packet = bd.getPacket())
        ids[p.dma].insert(packet->getPktId());
    });
  }
  for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
    std::optional<PacketInfoAttr> packet = alloc.getPacket();
    std::optional<TileID> tile = tileOf(alloc.getTile());
    if (packet && tile)
      ids[{*tile, alloc.getChannelDir(),
           static_cast<int>(alloc.getChannelIndex())}]
          .insert(packet->getPktId());
  }
  device.walk([&](AIEX::NpuDmaMemcpyNdOp memcpy) {
    std::optional<PacketInfoAttr> packet = memcpy.getPacket();
    if (!packet)
      return;
    if (std::optional<TileDMAChannel> key =
            channelOfSymbol(device, memcpy.getMetadata().getRootReference()))
      ids[*key].insert(packet->getPktId());
  });
  return ids;
}

class StreamTracer {
public:
  explicit StreamTracer(DeviceOp device)
      : maxPacketID(getTargetModel(device).getMaxPacketId()) {
    for (auto sb : device.getOps<SwitchboxOp>())
      switchboxes[sb.getTileOp().getTileID()] = sb;
    for (auto mux : device.getOps<ShimMuxOp>())
      shimMuxes[mux.getTileOp().getTileID()] = mux;
    sentIDs = collectSentPacketIDs(device);
  }

  std::vector<RoutedStream> trace() {
    for (auto &[tile, mux] : shimMuxes)
      for (Port p : inputPorts(mux.getConnections()))
        if (p.bundle != WireBundle::North)
          traceFrom({tile, p}, mux, p);
    for (auto &[tile, sb] : switchboxes)
      for (Port p : inputPorts(sb.getConnections()))
        if (!isDirectional(p.bundle))
          traceFrom({tile, p}, sb, p);
    return std::move(streams);
  }

private:
  struct Hop {
    Operation *interconnect;
    Port input;
  };

  static SmallVector<Port> inputPorts(Region &connections) {
    SmallVector<Port> ports;
    for (Operation &op : connections.front()) {
      std::optional<Port> p;
      if (auto connect = dyn_cast<ConnectOp>(op))
        p = connect.sourcePort();
      else if (auto rules = dyn_cast<PacketRulesOp>(op))
        p = rules.sourcePort();
      if (p && !llvm::is_contained(ports, *p))
        ports.push_back(*p);
    }
    return ports;
  }

  void traceFrom(StreamEndpoint src, Operation *interconnect, Port input) {
    std::optional<std::set<int>> ids;
    // A tile sends from its MM2S channel on the port of the same number.
    if (src.port.bundle == WireBundle::DMA) {
      auto it = sentIDs.find({src.tile, DMAChannelDir::MM2S, src.port.channel});
      if (it != sentIDs.end())
        ids = it->second;
    }
    if (!ids) {
      step(src, {interconnect, input}, std::nullopt);
      return;
    }
    for (int id : *ids)
      step(src, {interconnect, input}, id);
  }

  // The switchbox input a master port of `from` drives, or the tile port it
  // leaves the fabric at.
  std::variant<Hop, StreamEndpoint> follow(Operation *from, Port out) {
    TileID here = isa<SwitchboxOp>(from)
                      ? cast<SwitchboxOp>(from).getTileOp().getTileID()
                      : cast<ShimMuxOp>(from).getTileOp().getTileID();
    if (isa<ShimMuxOp>(from)) {
      if (out.bundle == WireBundle::North)
        if (auto sb = switchboxes.find(here); sb != switchboxes.end())
          return Hop{sb->second, {WireBundle::South, out.channel}};
      return StreamEndpoint{here, out};
    }
    if (!isDirectional(out.bundle))
      return StreamEndpoint{here, out};
    if (out.bundle == WireBundle::South)
      if (auto mux = shimMuxes.find(here); mux != shimMuxes.end())
        return Hop{mux->second, {WireBundle::North, out.channel}};
    TileID next = here;
    WireBundle in = WireBundle::South;
    switch (out.bundle) {
    case WireBundle::North:
      next.row++;
      in = WireBundle::South;
      break;
    case WireBundle::South:
      next.row--;
      in = WireBundle::North;
      break;
    case WireBundle::East:
      next.col++;
      in = WireBundle::West;
      break;
    default:
      next.col--;
      in = WireBundle::East;
      break;
    }
    if (auto sb = switchboxes.find(next); sb != switchboxes.end())
      return Hop{sb->second, {in, out.channel}};
    return StreamEndpoint{here, out};
  }

  // Follows the stream `src` sends with `id` on from `hop`, along every
  // branch that does not come back to an input it already took.
  void step(StreamEndpoint src, Hop hop, std::optional<int> id) {
    std::pair<Operation *, int> input{
        hop.interconnect,
        static_cast<int>(hop.input.bundle) * 1024 + hop.input.channel};
    if (!onPath.insert(input).second)
      return;
    llvm::scope_exit leave([&] { onPath.erase(input); });
    auto sb = dyn_cast<SwitchboxOp>(hop.interconnect);
    Region &connections =
        sb ? sb.getConnections()
           : cast<ShimMuxOp>(hop.interconnect).getConnections();
    Block &b = connections.front();

    auto next = [&](Port out, std::optional<int> nextID,
                    std::optional<int> arbiter, bool keepsPktHeader = false) {
      if (sb)
        path.push_back({sb.getTileOp().getTileID(), hop.input, arbiter});
      llvm::scope_exit backtrack([&] {
        if (sb)
          path.pop_back();
      });
      std::variant<Hop, StreamEndpoint> to = follow(hop.interconnect, out);
      if (auto *endpoint = std::get_if<StreamEndpoint>(&to))
        streams.push_back({src, *endpoint, nextID, keepsPktHeader, path});
      else
        step(src, std::get<Hop>(to), nextID);
    };

    for (auto connect : b.getOps<ConnectOp>())
      if (connect.sourcePort() == hop.input)
        next(connect.destPort(), id, std::nullopt);

    for (auto rules : b.getOps<PacketRulesOp>()) {
      if (rules.sourcePort() != hop.input)
        continue;
      auto route = [&](PacketRuleOp rule, int ruleID) {
        std::optional<int> arbiter;
        if (auto amsel = rule.getAmsel().getDefiningOp<AMSelOp>())
          arbiter = amsel.arbiterIndex();
        for (auto masterSet : b.getOps<MasterSetOp>())
          if (llvm::is_contained(masterSet.getAmsels(), rule.getAmsel()))
            next(masterSet.destPort(), ruleID, arbiter,
                 masterSet.getKeepPktHeader().value_or(false));
      };
      for (int packetID = id.value_or(0); packetID <= id.value_or(maxPacketID);
           ++packetID)
        for (auto rule : rules.getRules().front().getOps<PacketRuleOp>())
          if ((packetID & rule.maskInt()) ==
              (rule.valueInt() & rule.maskInt())) {
            route(rule, packetID);
            break;
          }
    }
  }

  std::map<TileID, SwitchboxOp> switchboxes;
  std::map<TileID, ShimMuxOp> shimMuxes;
  std::map<TileDMAChannel, std::set<int>> sentIDs;
  std::vector<RoutedStream> streams;
  // The interconnect inputs the stream `step` follows has taken, and the
  // switchboxes it has passed.
  llvm::DenseSet<std::pair<Operation *, int>> onPath;
  SmallVector<StreamHop, 8> path;
  int maxPacketID;
};

} // namespace

std::vector<RoutedStream> AIE::traceRoutedStreams(DeviceOp device) {
  return StreamTracer(device).trace();
}

std::vector<RoutedStream> AIE::requestedStreams(DeviceOp device) {
  std::vector<RoutedStream> streams;
  for (auto flow : device.getOps<FlowOp>()) {
    std::optional<TileID> srcTile = tileOf(flow.getSource());
    std::optional<TileID> dstTile = tileOf(flow.getDest());
    if (srcTile && dstTile)
      streams.push_back(
          {{*srcTile, {flow.getSourceBundle(), flow.sourceIndex()}},
           {*dstTile, {flow.getDestBundle(), flow.destIndex()}},
           std::nullopt});
  }
  for (auto flow : device.getOps<PacketFlowOp>()) {
    Block &b = flow.getPorts().front();
    for (auto src : b.getOps<PacketSourceOp>()) {
      std::optional<TileID> srcTile = tileOf(src.getTile());
      if (!srcTile)
        continue;
      for (auto dst : b.getOps<PacketDestOp>())
        if (std::optional<TileID> dstTile = tileOf(dst.getTile()))
          streams.push_back({{*srcTile, src.port()},
                             {*dstTile, dst.port()},
                             static_cast<int>(flow.IDInt()),
                             flow.getKeepPktHeader().value_or(false),
                             {},
                             static_cast<int>(flow.getMask().value_or(~0))});
    }
  }
  return streams;
}

StreamVolumeAnalysis::StreamVolumeAnalysis(DeviceOp device,
                                           ArrayRef<RoutedStream> streams)
    : device(device), streams(streams) {
  for (DmaChannelProgram &p : collectDmaPrograms(device))
    programs[p.dma].push_back(std::move(p));
  for (const auto &[dma, channelPrograms] : programs)
    for (const DmaChannelProgram &p : channelPrograms)
      forEachInProgram<UseLockOp>(
          p, [&](UseLockOp use) { lockUseProgram[use] = &p; });
  device.walk([&](AIEX::NpuDmaMemcpyNdOp memcpy) {
    if (std::optional<TileDMAChannel> dma =
            channelOfSymbol(device, memcpy.getMetadata().getRootReference()))
      memcpys[*dma].push_back(memcpy);
  });
}

// The BD blocks of a program in the order its channel runs them.
static SmallVector<Block *> chainBlocks(const DmaChannelProgram &p) {
  SmallVector<Block *> sequence(p.bds);
  for (Region *r : p.regions)
    sequence.push_back(&r->front());
  return sequence;
}

// The first block of the part of `sequence` a program runs over and over:
// where a looping chain returns to, or the whole chain for a repeated one.
static size_t cycleStart(const DmaChannelProgram &p) {
  return p.loops ? p.loopStart : 0;
}

// Index into `sequence` of the block a program runs at `step`: the blocks
// before the cycle once, then the cycle over and over.
static size_t blockAt(const DmaChannelProgram &p, size_t size, uint64_t step) {
  size_t start = cycleStart(p);
  if (step < start)
    return step;
  return start + (step - start) % (size - start);
}

// Whether the stream carries packets with id `id`.
static bool carriesID(const RoutedStream &s, int id) {
  return !s.packetID || ((id ^ *s.packetID) & s.packetMask) == 0;
}

// The value `use` acquires or releases, if it is a constant.
static std::optional<int64_t> lockAmount(UseLockOp use) {
  APInt amount;
  if (!use.getLock().getDefiningOp<LockOp>() ||
      !matchPattern(use.getValue(), m_ConstantInt(&amount)))
    return std::nullopt;
  return amount.getSExtValue();
}

// The tokens `use` takes or gives, if that is a known count.
static std::optional<uint64_t> lockTokens(UseLockOp use) {
  std::optional<int64_t> n = lockAmount(use);
  if (!n || *n < 0)
    return std::nullopt;
  return *n;
}

constexpr int maxBDSteps = 1024;

static bool inLoop(Operation *op) {
  return op->getParentOfType<LoopLikeOpInterface>() != nullptr;
}

std::optional<uint64_t>
StreamVolumeAnalysis::sendVolume(const RoutedStream &stream) const {
  if (stream.src.port.bundle != WireBundle::DMA)
    return std::nullopt;
  TileDMAChannel dma{stream.src.tile, DMAChannelDir::MM2S,
                     stream.src.port.channel};
  auto carries = [&](std::optional<PacketInfoAttr> packet) {
    return !packet || carriesID(stream, packet->getPktId());
  };
  constexpr uint64_t pktHeaderBytes = 4;
  auto headerBytes = [&](std::optional<PacketInfoAttr> packet) -> uint64_t {
    return packet && stream.keepsPktHeader ? pktHeaderBytes : 0;
  };
  auto bytesOf = [&](DMABDOp bd) -> uint64_t {
    if (!carries(bd.getPacket()))
      return 0;
    return bd.getLenInBytes() + headerBytes(bd.getPacket());
  };
  bool known = false;
  uint64_t total = 0;
  if (auto it = programs.find(dma); it != programs.end()) {
    for (const DmaChannelProgram &p : it->second) {
      bool carried = false;
      uint64_t bytes = 0;
      forEachInProgram<DMABDOp>(p, [&](DMABDOp bd) {
        if (!carries(bd.getPacket()))
          return;
        carried = true;
        bytes += bytesOf(bd);
      });
      known = true;
      if (!carried)
        continue;
      if (p.loops) {
        std::optional<uint64_t> looped = loopedVolume(p, bytesOf);
        if (!looped)
          return std::nullopt;
        total += *looped;
        continue;
      }
      uint64_t runs = p.passes;
      if (isa<AIEX::DMAConfigureTaskOp, AIEX::DMAConfigureTaskForOp>(p.op)) {
        auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(p.op);
        auto taskFor = dyn_cast<AIEX::DMAConfigureTaskForOp>(p.op);
        if ((task && task.getRepeatCountVal()) ||
            (taskFor && taskFor.getRepeatCountVal()))
          return std::nullopt;
        runs = 0;
        for (Operation *user : p.op->getUsers()) {
          if (isa<AIEX::DMAAwaitTaskOp, AIEX::DMAFreeTaskOp>(user))
            continue;
          // Any other user, such as an scf.yield or an scf.for init, can
          // forward the task to starts this does not count.
          if (!isa<AIEX::DMAStartTaskOp>(user))
            return std::nullopt;
          if (inLoop(user))
            return std::nullopt;
          runs += p.passes;
        }
      }
      total += bytes * runs;
    }
  }
  if (auto it = memcpys.find(dma); it != memcpys.end()) {
    for (Operation *op : it->second) {
      auto memcpy = cast<AIEX::NpuDmaMemcpyNdOp>(op);
      std::optional<PacketInfoAttr> packet = memcpy.getPacket();
      if (!packet)
        packet = ShimDMAAllocationOp::getForSymbol(
                     device, memcpy.getMetadata().getRootReference())
                     .getPacket();
      if (!carries(packet))
        continue;
      auto type = dyn_cast<BaseMemRefType>(memcpy.getMemref().getType());
      if (inLoop(memcpy) || !type || !type.getElementType().isIntOrFloat())
        return std::nullopt;
      uint64_t elements = 1;
      SmallVector<OpFoldResult> sizes = memcpy.getMixedSizes();
      for (OpFoldResult size : sizes) {
        std::optional<int64_t> n = getConstantIntValue(size);
        if (!n)
          return std::nullopt;
        elements *= *n;
      }
      // The outermost dimension re-runs the BD (as iterations, or as repeats
      // when its stride is 0), and each run sends its own header.
      uint64_t runs = *getConstantIntValue(sizes.front());
      known = true;
      total += elements * type.getElementTypeBitWidth() / 8 +
               runs * headerBytes(packet);
    }
  }
  if (!known)
    return std::nullopt;
  return total;
}

std::optional<uint64_t> StreamVolumeAnalysis::loopedVolume(
    const DmaChannelProgram &program,
    function_ref<uint64_t(DMABDOp)> bytesOf) const {
  uint64_t bytes = 0;
  if (!walkLoop(program, [&](DMABDOp bd) { bytes += bytesOf(bd); }))
    return std::nullopt;
  return bytes;
}

bool StreamVolumeAnalysis::walkLoop(const DmaChannelProgram &p,
                                    function_ref<void(DMABDOp)> visit) const {
  // AIE1 locks hold a state rather than count tokens.
  if (getTargetModel(device).getTargetArch() == AIEArch::AIE1 ||
      !visiting.insert(&p).second)
    return false;
  llvm::scope_exit done([&] { visiting.erase(&p); });
  SmallVector<Block *> sequence = chainBlocks(p);
  if (sequence.empty())
    return false;
  DenseMap<Operation *, uint64_t> tokens;
  for (int step = 0; step < maxBDSteps; step++) {
    for (Operation &bdOp : *sequence[blockAt(p, sequence.size(), step)]) {
      if (auto use = dyn_cast<UseLockOp>(bdOp)) {
        auto lock = use.getLock().getDefiningOp<LockOp>();
        std::optional<uint64_t> n = lockTokens(use);
        if (!n || (!use.release() && !use.acquireGE()))
          return false;
        auto it = tokens.find(lock);
        if (it == tokens.end()) {
          std::optional<uint64_t> others = tokensFromOthers(lock, &p);
          if (!others)
            return false;
          it = tokens.try_emplace(lock, lock.getInit().value_or(0) + *others)
                   .first;
        }
        if (use.release())
          it->second += *n;
        else if (it->second < *n)
          return true;
        else
          it->second -= *n;
      } else if (auto bd = dyn_cast<DMABDOp>(bdOp)) {
        visit(bd);
      }
    }
  }
  return false;
}

// Whether `op` runs at most once each time its core runs its body.
static bool onceInCore(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       op = parent, parent = parent->getParentOp()) {
    if (!op->getParentRegion()->hasOneBlock() ||
        isa<LoopLikeOpInterface>(parent))
      return false;
    if (isa<CoreOp>(parent))
      return true;
  }
  return false;
}

std::optional<uint64_t>
StreamVolumeAnalysis::tokensFromOthers(LockOp lock,
                                       const DmaChannelProgram *self) const {
  llvm::SetVector<const DmaChannelProgram *> releasers;
  uint64_t tokens = 0;
  for (Operation *user : lock->getUsers()) {
    auto use = dyn_cast<UseLockOp>(user);
    if (!use)
      return std::nullopt;
    if (!use.release())
      continue;
    auto owner = lockUseProgram.find(use);
    if (owner == lockUseProgram.end()) {
      std::optional<uint64_t> n = lockTokens(use);
      if (!onceInCore(use) || !n)
        return std::nullopt;
      tokens += *n;
      continue;
    }
    if (owner->second != self)
      releasers.insert(owner->second);
  }
  for (const DmaChannelProgram *program : releasers) {
    std::optional<uint64_t> n = releasesOver(*program, lock);
    if (!n)
      return std::nullopt;
    tokens += *n;
  }
  return tokens;
}

std::optional<bool>
StreamVolumeAnalysis::maySendAfter(const RoutedStream &first,
                                   const RoutedStream &then) const {
  if (first.src != then.src)
    return std::nullopt;
  if (!first.packetID || !then.packetID ||
      ((*first.packetID ^ *then.packetID) & first.packetMask &
       then.packetMask) == 0)
    return true;
  if (first.src.port.bundle != WireBundle::DMA)
    return std::nullopt;
  TileDMAChannel dma{first.src.tile, DMAChannelDir::MM2S,
                     first.src.port.channel};
  auto it = programs.find(dma);
  if (it == programs.end() || it->second.size() != 1 ||
      !isa<DMAStartOp, DMAOp>(it->second.front().op) || memcpys.count(dma))
    return std::nullopt;
  const DmaChannelProgram &p = it->second.front();
  auto carries = [](DMABDOp bd, const RoutedStream &s) {
    std::optional<PacketInfoAttr> packet = bd.getPacket();
    return !packet || carriesID(s, packet->getPktId());
  };
  bool sent = false, after = false;
  auto visit = [&](DMABDOp bd) {
    after |= sent && carries(bd, then);
    sent |= carries(bd, first);
  };
  if (p.loops) {
    if (!walkLoop(p, visit))
      return std::nullopt;
    return after;
  }
  SmallVector<Block *> sequence = chainBlocks(p);
  for (uint64_t pass = 0; pass < p.passes && !after; pass++)
    for (Block *b : sequence)
      for (DMABDOp bd : b->getOps<DMABDOp>())
        visit(bd);
  return after;
}

std::optional<uint64_t>
StreamVolumeAnalysis::releasesOver(const DmaChannelProgram &p,
                                   LockOp lock) const {
  if (!isa<DMAStartOp, DMAOp>(p.op))
    return std::nullopt;
  SmallVector<Block *> sequence = chainBlocks(p);
  if (!p.loops) {
    uint64_t perPass = 0;
    for (Block *b : sequence)
      for (auto use : b->getOps<UseLockOp>())
        if (use.release() && use.getLock().getDefiningOp() == lock) {
          std::optional<uint64_t> n = lockTokens(use);
          if (!n)
            return std::nullopt;
          perPass += *n;
        }
    return perPass * p.passes;
  }
  // A looping receiver finishes only the BDs what it is sent fills.
  if (p.dma.dir != DMAChannelDir::S2MM || sequence.empty())
    return std::nullopt;
  StreamEndpoint endpoint{p.dma.tile, {WireBundle::DMA, p.dma.channel}};
  uint64_t received = 0;
  for (const RoutedStream &s : streams) {
    if (s.dst != endpoint)
      continue;
    std::optional<uint64_t> bytes = sendVolume(s);
    if (!bytes)
      return std::nullopt;
    received += *bytes;
  }
  uint64_t filled = 0, tokens = 0;
  for (int step = 0; step < maxBDSteps; step++) {
    for (Operation &bdOp : *sequence[blockAt(p, sequence.size(), step)]) {
      if (auto bd = dyn_cast<DMABDOp>(bdOp)) {
        if (filled + bd.getLenInBytes() > received)
          return tokens;
        filled += bd.getLenInBytes();
      } else if (auto use = dyn_cast<UseLockOp>(bdOp)) {
        if (!use.release() || use.getLock().getDefiningOp() != lock)
          continue;
        std::optional<uint64_t> n = lockTokens(use);
        if (!n)
          return std::nullopt;
        tokens += *n;
      }
    }
  }
  return std::nullopt;
}

static std::optional<uint64_t> programCapacity(const DmaChannelProgram &p,
                                               DeviceOp device) {
  if (!isa<DMAStartOp, DMAOp>(p.op))
    return 0;

  // Run the BD chain from its initial lock values until an acquire blocks.
  SmallVector<Block *> sequence = chainBlocks(p);
  if (sequence.empty())
    return 0;
  // AIE1 locks hold a state an acquire waits to equal and a release sets; later
  // locks count, and an acquire waiting for an exact count is taken to block.
  bool stateLocks = getTargetModel(device).getTargetArch() == AIEArch::AIE1;
  DenseMap<Operation *, int64_t> lockValues;
  uint64_t bytes = 0;
  // Lock values and bytes when the cycle last began. A cycle that leaves every
  // lock where it found it, or with more tokens, runs again the same way.
  std::optional<DenseMap<Operation *, int64_t>> cycleLocks;
  uint64_t cycleBytes = 0;
  size_t start = cycleStart(p);
  for (uint64_t step = 0;; step++) {
    if (!p.loops && step >= p.passes * sequence.size())
      return bytes;
    size_t i = blockAt(p, sequence.size(), step);
    if (i == start) {
      if (cycleLocks && cycleLocks->size() == lockValues.size() &&
          llvm::all_of(lockValues, [&](const auto &lv) {
            int64_t before = cycleLocks->at(lv.first);
            return stateLocks ? lv.second == before : lv.second >= before;
          })) {
        if (p.loops)
          return std::nullopt;
        return bytes +
               (p.passes - step / sequence.size()) * (bytes - cycleBytes);
      }
      cycleLocks = lockValues;
      cycleBytes = bytes;
    }
    // Past the analysis limit, what it took in so far is a safe capacity.
    if (step >= static_cast<uint64_t>(maxBDSteps))
      return bytes;
    for (Operation &bdOp : *sequence[i]) {
      if (auto use = dyn_cast<UseLockOp>(bdOp)) {
        std::optional<int64_t> n = lockAmount(use);
        if (!n)
          return 0;
        auto lock = use.getLock().getDefiningOp<LockOp>();
        int64_t &value =
            lockValues.try_emplace(lock, lock.getInit().value_or(0))
                .first->second;
        if (stateLocks) {
          if (use.release())
            value = *n;
          else if (value != *n)
            return bytes;
        } else if (use.release()) {
          value += *n;
        } else if (!use.acquireGE() || value < *n) {
          return bytes;
        } else {
          value -= *n;
        }
      } else if (auto bd = dyn_cast<DMABDOp>(bdOp)) {
        bytes += bd.getLenInBytes();
      }
    }
  }
}

std::optional<uint64_t>
StreamVolumeAnalysis::receiveCapacity(const StreamEndpoint &endpoint) const {
  if (endpoint.port.bundle != WireBundle::DMA)
    return 0;
  auto it = programs.find(
      {endpoint.tile, DMAChannelDir::S2MM, endpoint.port.channel});
  if (it == programs.end())
    return 0;
  std::optional<uint64_t> capacity;
  for (const DmaChannelProgram &p : it->second)
    if (std::optional<uint64_t> c = programCapacity(p, device))
      capacity = std::min(*c, capacity.value_or(*c));
  return capacity;
}

bool StreamVolumeAnalysis::canFill(const StreamEndpoint &endpoint,
                                   ArrayRef<RoutedStream> streams) const {
  std::optional<uint64_t> capacity = receiveCapacity(endpoint);
  if (!capacity)
    return false;
  uint64_t sent = 0;
  for (const RoutedStream &s : streams) {
    if (s.dst != endpoint)
      continue;
    std::optional<uint64_t> bytes = sendVolume(s);
    if (!bytes)
      return true;
    sent += *bytes;
  }
  return sent > *capacity;
}

StreamWaitGraph::StreamWaitGraph(DeviceOp device,
                                 ArrayRef<RoutedStream> streams,
                                 const StreamVolumeAnalysis &volumes) {
  std::set<TileDMAChannel> neverFull;
  for (const RoutedStream &s : streams)
    if (s.dst.port.bundle == WireBundle::DMA &&
        !volumes.canFill(s.dst, streams))
      neverFull.insert({s.dst.tile, DMAChannelDir::S2MM, s.dst.port.channel});
  auto waitsOnLocks = [&](unsigned agent) {
    const Agent &a = agents[agent];
    return a.isCore || !neverFull.count(a.dma);
  };

  // Who acquires and who releases each lock.
  DenseMap<Operation *, llvm::SetVector<unsigned>> acquirers, releasers;
  auto noteLock = [&](UseLockOp use, unsigned agent) {
    Operation *lock = use.getLock().getDefiningOp();
    if (!lock)
      return;
    if (use.getAction() == LockAction::Release)
      releasers[lock].insert(agent);
    else
      acquirers[lock].insert(agent);
  };

  // Agents whose waits the design spells out; see the end of this constructor
  // for the rest.
  for (auto core : device.getOps<CoreOp>()) {
    unsigned agent = getOrCreate(Agent::core(core.getTileOp().getTileID()));
    modeled.insert(agent);
    core.walk([&](UseLockOp use) { noteLock(use, agent); });
  }
  for (const DmaChannelProgram &p : collectDmaPrograms(device)) {
    unsigned agent = getOrCreate(Agent::channel(p.dma));
    modeled.insert(agent);
    forEachInProgram<UseLockOp>(p,
                                [&](UseLockOp use) { noteLock(use, agent); });
  }
  // Document order, not pointer order: edge order picks the chain a
  // diagnostic reports.
  for (auto lock : device.getOps<LockOp>()) {
    auto waiting = acquirers.find(lock), it = releasers.find(lock);
    if (waiting == acquirers.end() || it == releasers.end())
      continue;
    for (unsigned p : waiting->second)
      for (unsigned q : it->second)
        if (p != q && waitsOnLocks(p))
          addEdge(p, q, EdgeKind::Lock);
  }

  for (const RoutedStream &s : streams) {
    std::optional<unsigned> from, to;
    if (std::optional<Agent> agent = Agent::at(s.src, true))
      from = getOrCreate(*agent);
    if (std::optional<Agent> agent = Agent::at(s.dst, false))
      to = getOrCreate(*agent);
    if (!from || !to || *from == *to)
      continue;
    addEdge(*from, *to, EdgeKind::Stream);
    addEdge(*to, *from, EdgeKind::Stream);
  }

  // The host issues a channel's transfer only after every wait before it in
  // the runtime sequence completes. Each issue is modeled: a later one can
  // follow waits the first did not. A wait on a channel not known here may be
  // on any the sequence issues, and an issue on one, or a raw register write,
  // may start any channel.
  unsigned numStreamAgents = agents.size();
  for (auto sequence : device.getOps<RuntimeSequenceOp>()) {
    auto bySymbol = [&](SymbolRefAttr symbol) -> SmallVector<TileDMAChannel> {
      if (std::optional<TileDMAChannel> key =
              channelOfSymbol(device, symbol.getRootReference()))
        return {*key};
      return {};
    };
    auto issued =
        [&](Operation *op) -> std::optional<SmallVector<TileDMAChannel>> {
      if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op))
        return bySymbol(memcpy.getMetadata());
      if (auto start = dyn_cast<AIEX::DMAStartTaskOp>(op))
        return channelsOfTask(device, start.getTask());
      if (isa<AIEX::DMAStartBdChainOp, AIEX::DMAStartBdChainForOp>(op))
        return channelsOfTask(device, op->getResult(0));
      if (auto push = dyn_cast<AIEX::NpuPushQueueOp>(op))
        return SmallVector<TileDMAChannel>{
            {{static_cast<int>(push.getColumn()),
              static_cast<int>(push.getRow())},
             push.getDirection(),
             static_cast<int>(push.getChannel())}};
      if (isa<AIEX::NpuWrite32Op, AIEX::NpuMaskWrite32Op,
              AIEX::NpuBlockWriteOp>(op))
        return SmallVector<TileDMAChannel>{};
      return std::nullopt;
    };
    auto awaited =
        [&](Operation *op) -> std::optional<SmallVector<TileDMAChannel>> {
      if (auto dmaWait = dyn_cast<AIEX::NpuDmaWaitOp>(op))
        return bySymbol(dmaWait.getSymbolAttr());
      if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        return channelsOfTask(device, await.getTask());
      if (auto sync = dyn_cast<AIEX::NpuSyncOp>(op)) {
        std::optional<int64_t> col = getConstantIntValue(sync.getColumn()),
                               row = getConstantIntValue(sync.getRow()),
                               dir = getConstantIntValue(sync.getDirection()),
                               channel = getConstantIntValue(sync.getChannel()),
                               cols = getConstantIntValue(sync.getColumnNum()),
                               rows = getConstantIntValue(sync.getRowNum());
        if (!col || !row || !dir || !channel || !cols || !rows)
          return SmallVector<TileDMAChannel>{};
        SmallVector<TileDMAChannel> keys;
        for (int64_t c = *col; c < *col + *cols; c++)
          for (int64_t r = *row; r < *row + *rows; r++)
            keys.push_back({{static_cast<int>(c), static_cast<int>(r)},
                            *dir ? DMAChannelDir::MM2S : DMAChannelDir::S2MM,
                            static_cast<int>(*channel)});
        return keys;
      }
      if (isa<AIEX::NpuMaskPollOp>(op))
        return SmallVector<TileDMAChannel>{};
      return std::nullopt;
    };
    auto agentOf = [&](const TileDMAChannel &dma) {
      return getOrCreate(Agent::channel(dma));
    };

    llvm::SetVector<TileDMAChannel, SmallVector<TileDMAChannel>,
                    std::set<TileDMAChannel>>
        issuedKeys;
    sequence.walk([&](Operation *op) {
      if (std::optional<SmallVector<TileDMAChannel>> keys = issued(op))
        issuedKeys.insert(keys->begin(), keys->end());
    });

    llvm::SetVector<unsigned> waited;
    walkLoopsTwice(sequence.getBody(), [&](Operation *op) {
      if (std::optional<SmallVector<TileDMAChannel>> keys = issued(op)) {
        for (const TileDMAChannel &key : *keys) {
          unsigned agent = agentOf(key);
          modeled.insert(agent);
          for (unsigned w : waited)
            if (w != agent)
              addEdge(agent, w, EdgeKind::Host);
        }
        if (keys->empty())
          for (unsigned a = 0; a < numStreamAgents; a++)
            if (!agents[a].isCore)
              for (unsigned w : waited)
                if (w != a)
                  addEdge(a, w, EdgeKind::Host);
      } else if (std::optional<SmallVector<TileDMAChannel>> keys =
                     awaited(op)) {
        for (const TileDMAChannel &key :
             keys->empty() ? issuedKeys.getArrayRef() : ArrayRef(*keys))
          waited.insert(agentOf(key));
      }
    });
  }

  // A channel nothing programs here is programmed elsewhere, in ways this
  // cannot see, so it may wait on anything else on its tile. One on a shim
  // tile is driven by the host, which may wait on any other shim tile first.
  const AIETargetModel &targetModel = getTargetModel(device);
  for (Agent &a : agents)
    a.onShim = !a.isCore &&
               targetModel.isShimNOCorPLTile(a.dma.tile.col, a.dma.tile.row);
  for (unsigned a = 0; a < agents.size(); a++) {
    if (modeled.contains(a) || !waitsOnLocks(a))
      continue;
    for (unsigned b = 0; b < agents.size(); b++)
      if (b != a && (agents[b].dma.tile == agents[a].dma.tile ||
                     (agents[a].onShim && agents[b].onShim)))
        addEdge(a, b, EdgeKind::Lock);
  }

  LLVM_DEBUG({
    for (unsigned a = 0; a < agents.size(); a++)
      for (const Edge &e : edges[a])
        llvm::dbgs() << "Wait: " << describe(a) << " on " << describe(e.to)
                     << (e.kind == EdgeKind::Lock     ? " (lock)"
                         : e.kind == EdgeKind::Stream ? " (stream)"
                                                      : " (host)")
                     << (e.kind == EdgeKind::Lock && !modeled.contains(a)
                             ? ", assumed"
                             : "")
                     << "\n";
  });
}

std::optional<StreamWaitGraph::Agent>
StreamWaitGraph::Agent::at(const StreamEndpoint &endpoint, bool sending) {
  if (endpoint.port.bundle == WireBundle::Core)
    return core(endpoint.tile);
  if (endpoint.port.bundle == WireBundle::DMA)
    return channel({endpoint.tile,
                    sending ? DMAChannelDir::MM2S : DMAChannelDir::S2MM,
                    endpoint.port.channel});
  return std::nullopt;
}

unsigned StreamWaitGraph::getOrCreate(const Agent &agent) {
  auto [it, inserted] =
      agentIDs.try_emplace({agent.isCore, agent.dma}, agents.size());
  if (inserted) {
    agents.push_back(agent);
    edges.emplace_back();
  }
  return it->second;
}

std::optional<unsigned> StreamWaitGraph::lookup(const Agent &agent) const {
  auto it = agentIDs.find({agent.isCore, agent.dma});
  if (it == agentIDs.end())
    return std::nullopt;
  return it->second;
}

void StreamWaitGraph::addEdge(unsigned from, unsigned to, EdgeKind kind) {
  if (llvm::none_of(edges[from], [&](const Edge &e) {
        return e.to == to && e.kind == kind;
      }))
    edges[from].push_back({to, kind});
}

std::optional<unsigned> StreamWaitGraph::agentAt(const StreamEndpoint &endpoint,
                                                 bool sending) const {
  if (std::optional<Agent> agent = Agent::at(endpoint, sending))
    return lookup(*agent);
  return std::nullopt;
}

SmallVector<unsigned> StreamWaitGraph::drainersOf(unsigned agent) const {
  if (agents[agent].isCore)
    return {agent};
  SmallVector<unsigned> drainers;
  for (const Edge &e : edges[agent])
    if (e.kind != EdgeKind::Stream)
      drainers.push_back(e.to);
  return drainers;
}

bool StreamWaitGraph::reaches(ArrayRef<unsigned> from,
                              ArrayRef<unsigned> targets,
                              ArrayRef<unsigned> avoid) const {
  return !waitChain(from, targets, avoid).empty();
}

SmallVector<unsigned>
StreamWaitGraph::waitChain(ArrayRef<unsigned> from, ArrayRef<unsigned> targets,
                           ArrayRef<unsigned> avoid) const {
  llvm::DenseMap<unsigned, std::optional<unsigned>> parent;
  for (unsigned a : avoid)
    parent.try_emplace(a, std::nullopt);
  std::deque<unsigned> worklist;
  for (unsigned a : from)
    if (parent.try_emplace(a, std::nullopt).second)
      worklist.push_back(a);
  while (!worklist.empty()) {
    unsigned a = worklist.front();
    worklist.pop_front();
    if (llvm::is_contained(targets, a)) {
      SmallVector<unsigned> chain;
      for (std::optional<unsigned> at = a; at; at = parent.lookup(*at))
        chain.push_back(*at);
      std::reverse(chain.begin(), chain.end());
      return chain;
    }
    for (const Edge &e : edges[a])
      if (parent.try_emplace(e.to, a).second)
        worklist.push_back(e.to);
  }
  return {};
}

std::string StreamWaitGraph::describe(unsigned id) const {
  const Agent &a = agents[id];
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "(" << a.dma.tile.col << ", " << a.dma.tile.row << ") ";
  if (a.isCore)
    os << "core";
  else
    os << stringifyDMAChannelDir(a.dma.dir) << " " << a.dma.channel;
  return s;
}

StreamDeadlockAnalysis::StreamDeadlockAnalysis(DeviceOp device,
                                               ArrayRef<RoutedStream> streams)
    : streams(streams), volumes(device, streams),
      graph(device, streams, volumes), stalls(streams.size()),
      silence(streams.size()) {}

bool StreamDeadlockAnalysis::canStall(size_t f) const {
  std::optional<bool> &stall = stalls[f];
  if (!stall)
    stall = volumes.canFill(streams[f].dst, streams);
  return *stall;
}

SmallVector<unsigned> StreamDeadlockAnalysis::blockingChain(size_t f,
                                                            size_t g) const {
  const RoutedStream &fs = streams[f], &gs = streams[g];
  std::optional<unsigned> fDst = graph.agentAt(fs.dst, false);
  if (!fDst)
    return {};
  SmallVector<unsigned> own;
  if (std::optional<unsigned> fSrc = graph.agentAt(fs.src, true))
    own.push_back(*fSrc);
  own.push_back(*fDst);
  SmallVector<unsigned> drainers = graph.drainersOf(*fDst);
  SmallVector<unsigned> avoid, targets;
  for (unsigned a : own)
    if (!llvm::is_contained(drainers, a))
      avoid.push_back(a);
  for (std::optional<unsigned> a :
       {graph.agentAt(gs.src, true), graph.agentAt(gs.dst, false)})
    if (a && !llvm::is_contained(own, *a))
      targets.push_back(*a);
  if (targets.empty())
    return {};
  return graph.waitChain(drainers, targets, avoid);
}

bool StreamDeadlockAnalysis::canBlock(size_t f, size_t g) const {
  if (auto it = blocks.find({f, g}); it != blocks.end())
    return it->second;
  return blocks[{f, g}] = canStall(f) && !silent(f) && !silent(g) &&
                          volumes.maySendAfter(streams[f], streams[g]) !=
                              std::optional(false) &&
                          !blockingChain(f, g).empty();
}

bool StreamDeadlockAnalysis::silent(size_t f) const {
  std::optional<bool> &quiet = silence[f];
  if (!quiet)
    quiet = volumes.sendVolume(streams[f]) == std::optional<uint64_t>(0);
  return *quiet;
}

std::string StreamDeadlockAnalysis::explainBlock(size_t f, size_t g) const {
  const RoutedStream &fs = streams[f], &gs = streams[g];
  SmallVector<unsigned> chain = blockingChain(f, g);
  std::string s;
  llvm::raw_string_ostream os(s);
  os << describeStream(fs) << " can fill its receiver, and draining that "
     << "waits on ";
  for (auto [i, a] : llvm::enumerate(chain))
    os << (i ? ", then " : "") << graph.describe(a);
  os << (graph.agentAt(gs.dst, false) == chain.back() ? ", which receives "
                                                      : ", which sends ")
     << describeStream(gs) << '.';
  for (const std::string &a : assumptions(f, g))
    os << ' ' << a;
  return s;
}

SmallVector<std::string> StreamDeadlockAnalysis::assumptions(size_t f,
                                                             size_t g) const {
  const RoutedStream &fs = streams[f], &gs = streams[g];
  SmallVector<std::string> assumed;
  for (const RoutedStream &other : streams)
    if (other.dst == fs.dst && !volumes.sendVolume(other)) {
      assumed.push_back("The volume " + describeStream(other) +
                        " carries is unknown, so it is assumed to overrun its "
                        "receiver.");
      break;
    }
  if (fs.src == gs.src && !volumes.maySendAfter(fs, gs))
    assumed.push_back("Both come from " +
                      describeTilePort(fs.src.tile, fs.src.port) +
                      ", and the order it sends in is not modeled.");
  SmallVector<unsigned> chain = blockingChain(f, g);
  std::optional<unsigned> receiver = graph.agentAt(fs.dst, false);
  assert(receiver && "a stream that can block has a receiver");
  SmallVector<unsigned> waiters{*receiver};
  waiters.append(chain.begin(), std::prev(chain.end()));
  for (auto [i, a] : llvm::enumerate(waiters))
    if (!graph.isModeled(a) &&
        !llvm::is_contained(ArrayRef(waiters).take_front(i), a))
      assumed.push_back(
          "Nothing in the design programs " + graph.describe(a) +
          ", so it is assumed to wait on anything on its tile" +
          (graph.getAgent(a).onShim ? " or on another shim tile." : "."));
  return assumed;
}

std::string AIE::describeStream(const RoutedStream &stream) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << (stream.packetID ? "packet flow " : "flow ")
     << describeTilePort(stream.src.tile, stream.src.port) << " -> "
     << describeTilePort(stream.dst.tile, stream.dst.port);
  if (stream.packetID)
    os << " (id " << *stream.packetID << ")";
  return s;
}

StreamConflicts::StreamConflicts(DeviceOp device)
    : device(device), streams(requestedStreams(device)),
      numRequested(streams.size()) {
  for (RoutedStream &s : traceRoutedStreams(device))
    streams.push_back(std::move(s));
  std::map<TreeKey, size_t> treeIDs;
  for (size_t i = 0; i < streams.size(); i++) {
    size_t tree = treeMembers.size();
    if (std::optional<TreeKey> key = treeKey(i))
      tree = treeIDs.try_emplace(*key, tree).first->second;
    if (tree == treeMembers.size())
      treeMembers.emplace_back();
    treeMembers[tree].push_back(i);
    treeOf.push_back(tree);
  }
  waits.resize(treeMembers.size());
}

std::optional<StreamConflicts::TreeKey>
StreamConflicts::treeKey(size_t s) const {
  const RoutedStream &stream = streams[s];
  if (!stream.packetID)
    return std::nullopt;
  return TreeKey{stream.src.tile, stream.src.port, *stream.packetID,
                 s < numRequested};
}

const StreamDeadlockAnalysis &StreamConflicts::getAnalysis() const {
  if (!analysis)
    analysis.emplace(device, streams);
  return *analysis;
}

bool StreamConflicts::blocks(size_t s, size_t t) const {
  const RoutedStream &a = streams[s], &b = streams[t];
  if (a.src == b.src || a.dst == b.dst)
    return false;
  return getAnalysis().canBlock(s, t);
}

// Trees from one source, or into one receiver, already wait on each other
// there whatever the routing.
bool StreamConflicts::related(size_t s, size_t t) const {
  if (streams[s].src == streams[t].src)
    return true;
  for (size_t m : treeMembers[treeOf[s]])
    for (size_t n : treeMembers[treeOf[t]])
      if (streams[m].dst == streams[n].dst)
        return true;
  return false;
}

bool StreamConflicts::conflict(size_t s, size_t t) const {
  return !related(s, t) && (blocks(s, t) || blocks(t, s));
}

// The chains holdCycle follows from a tree stuck at a receiver to the trees
// it waits on, which hold for any routing.
const DenseMap<size_t, std::pair<size_t, size_t>> &
StreamConflicts::waitsFrom(size_t a) const {
  std::optional<DenseMap<size_t, std::pair<size_t, size_t>>> &cached = waits[a];
  if (cached)
    return *cached;
  DenseMap<size_t, std::pair<size_t, size_t>> &reached = cached.emplace();
  std::deque<size_t> work{a};
  while (!work.empty()) {
    size_t u = work.front();
    work.pop_front();
    for (size_t g = 0; g < treeMembers.size(); g++) {
      if (g == a || reached.contains(g) ||
          !streams[treeMembers[g].front()].packetID)
        continue;
      auto by = [&]() -> std::optional<std::pair<size_t, size_t>> {
        for (size_t x : treeMembers[u])
          for (size_t m : treeMembers[g])
            if (blocks(x, m))
              return std::pair{x, m};
        return std::nullopt;
      }();
      if (!by)
        continue;
      reached[g] = *by;
      work.push_back(g);
    }
  }
  return reached;
}

bool StreamConflicts::mustSeparate(size_t s, size_t t) const {
  if (related(s, t))
    return false;
  return blocks(s, t) || blocks(t, s) ||
         waitsFrom(treeOf[s]).contains(treeOf[t]) ||
         waitsFrom(treeOf[t]).contains(treeOf[s]);
}

std::string StreamConflicts::explain(size_t s, size_t t) const {
  const StreamDeadlockAnalysis &a = getAnalysis();
  if (a.canBlock(s, t))
    return a.explainBlock(s, t);
  if (a.canBlock(t, s))
    return a.explainBlock(t, s);
  if (!waitsFrom(treeOf[s]).contains(treeOf[t]))
    std::swap(s, t);
  SmallVector<std::string> steps;
  for (size_t g = treeOf[t]; g != treeOf[s];) {
    auto [x, m] = waitsFrom(treeOf[s]).at(g);
    steps.push_back(a.explainBlock(x, m));
    g = treeOf[x];
  }
  std::reverse(steps.begin(), steps.end());
  return llvm::join(steps, " ");
}

SmallVector<std::pair<size_t, size_t>> StreamConflicts::unavoidable() const {
  SmallVector<std::pair<size_t, size_t>> pairs;
  const StreamDeadlockAnalysis &a = getAnalysis();
  for (size_t s = 0; s < numRequested; s++)
    for (size_t t = 0; t < numRequested; t++) {
      if (s == t || !(streams[s].packetID || streams[t].packetID) ||
          !related(s, t) || !a.canBlock(s, t))
        continue;
      if (a.assumptions(s, t).empty())
        pairs.push_back({s, t});
      else
        LLVM_DEBUG(llvm::dbgs() << "Unavoidable only by assumption: "
                                << a.explainBlock(s, t) << "\n");
    }
  return pairs;
}

namespace {
// A wait in the graph holdCycle searches: a head stuck at one node waits on
// one at `to`, for the reason `step` gives, if any.
struct WaitEdge {
  size_t to;
  std::optional<HoldCycle::Step> step;
};
// The graph holdCycle searches, built as it is walked. Node `entry` leads to
// every root.
struct WaitGraph {
  llvm::function_ref<ArrayRef<WaitEdge>(size_t)> successors;
  size_t entry;
};
using WaitNode = std::pair<const WaitGraph *, size_t>;
} // namespace

namespace llvm {
template <>
struct GraphTraits<const WaitGraph *> {
  using NodeRef = WaitNode;
  struct ToNode {
    const WaitGraph *graph;
    NodeRef operator()(const WaitEdge &e) const { return {graph, e.to}; }
  };
  using ChildIteratorType = mapped_iterator<const WaitEdge *, ToNode>;
  static NodeRef getEntryNode(const WaitGraph *g) { return {g, g->entry}; }
  static ChildIteratorType child_begin(NodeRef n) {
    return map_iterator(n.first->successors(n.second).begin(), ToNode{n.first});
  }
  static ChildIteratorType child_end(NodeRef n) {
    return map_iterator(n.first->successors(n.second).end(), ToNode{n.first});
  }
};
} // namespace llvm

std::optional<HoldCycle>
StreamConflicts::holdCycle(ArrayRef<SmallVector<StreamHop, 8>> routes) const {
  // The packets one source sends with one id move down every branch as one.
  struct Tree {
    SmallVector<size_t, 2> members;
    SmallVector<std::pair<TileID, Port>, 8> hops;
    SmallVector<int, 8> parent;
    SmallVector<std::optional<int>, 8> arbiter;
    size_t base = 0;
  };
  std::vector<Tree> trees;
  std::map<TreeKey, size_t> treeIDs;
  for (size_t i = 0; i < streams.size(); i++) {
    std::optional<TreeKey> key = treeKey(i);
    if (!key || getAnalysis().silent(i))
      continue;
    auto [it, inserted] = treeIDs.try_emplace(*key, trees.size());
    if (inserted)
      trees.emplace_back();
    Tree &tree = trees[it->second];
    tree.members.push_back(i);
    int prev = -1;
    for (const StreamHop &hop : routes[i]) {
      std::pair<TileID, Port> key{hop.tile, hop.input};
      int h = llvm::find(tree.hops, key) - tree.hops.begin();
      if (h == static_cast<int>(tree.hops.size())) {
        tree.hops.push_back(key);
        tree.parent.push_back(prev);
        tree.arbiter.push_back(hop.arbiter);
      }
      prev = h;
    }
  }

  auto related = [&](size_t a, size_t b) {
    return this->related(trees[a].members.front(), trees[b].members.front());
  };

  // Per tree, a node for a head of it stuck entering each hop or at each
  // receiver, one for it stuck anywhere, and one per hop for it stuck
  // anywhere past that hop, so still holding what it took there.
  enum class Kind { Hop, Receiver, Anywhere, Past };
  struct Node {
    size_t tree;
    Kind kind;
    size_t index;
  };
  std::vector<Node> nodes;
  for (size_t t = 0; t < trees.size(); t++) {
    Tree &tree = trees[t];
    tree.base = nodes.size();
    for (size_t h = 0; h < tree.hops.size(); h++)
      nodes.push_back({t, Kind::Hop, h});
    for (size_t m = 0; m < tree.members.size(); m++)
      nodes.push_back({t, Kind::Receiver, m});
    nodes.push_back({t, Kind::Anywhere, 0});
    for (size_t h = 0; h < tree.hops.size(); h++)
      nodes.push_back({t, Kind::Past, h});
  }
  auto receiverNode = [&](size_t t, size_t m) {
    return trees[t].base + trees[t].hops.size() + m;
  };
  auto anywhereNode = [&](size_t t) {
    return receiverNode(t, trees[t].members.size());
  };
  auto pastNode = [&](size_t t, size_t h) { return anywhereNode(t) + 1 + h; };

  std::map<std::pair<TileID, Port>, SmallVector<std::pair<size_t, size_t>, 4>>
      entering;
  std::map<TileID, SmallVector<std::pair<size_t, size_t>, 8>> passing;
  for (size_t t = 0; t < trees.size(); t++)
    for (size_t h = 0; h < trees[t].hops.size(); h++) {
      entering[trees[t].hops[h]].push_back({t, h});
      passing[trees[t].hops[h].first].push_back({t, h});
    }

  using Edge = WaitEdge;
  SmallVector<size_t> roots;
  std::vector<std::optional<SmallVector<Edge, 4>>> edges(nodes.size() + 1);
  auto successors = [&](size_t n) -> ArrayRef<Edge> {
    if (edges[n])
      return *edges[n];
    SmallVector<Edge, 4> out;
    if (n == nodes.size()) {
      for (size_t root : roots)
        out.push_back({root, std::nullopt});
      return *(edges[n] = std::move(out));
    }
    const Node &node = nodes[n];
    const Tree &u = trees[node.tree];
    switch (node.kind) {
    case Kind::Hop: {
      auto [tile, input] = u.hops[node.index];
      size_t waiting = u.members.front();
      for (auto [t, ht] : entering.at({tile, input})) {
        if (t != node.tree && related(node.tree, t))
          continue;
        size_t sharer = trees[t].members.front();
        if (t != node.tree)
          out.push_back({pastNode(t, ht),
                         HoldCycle::Step{HoldCycle::Wait::Link, waiting, sharer,
                                         sharer, tile, input, input, -1}});
        std::optional<int> arbiter = trees[t].arbiter[ht];
        if (!arbiter)
          continue;
        for (auto [v, hv] : passing.at(tile)) {
          Port holderInput = trees[v].hops[hv].second;
          if (v == node.tree || v == t || holderInput == input ||
              trees[v].arbiter[hv] != arbiter || related(t, v))
            continue;
          out.push_back({pastNode(v, hv),
                         HoldCycle::Step{HoldCycle::Wait::Arbiter, waiting,
                                         sharer, trees[v].members.front(), tile,
                                         input, holderInput, *arbiter}});
        }
      }
      break;
    }
    case Kind::Receiver: {
      size_t s = u.members[node.index];
      for (size_t g = 0; g < trees.size(); g++) {
        if (g == node.tree)
          continue;
        for (size_t m : trees[g].members)
          if (blocks(s, m)) {
            out.push_back({anywhereNode(g),
                           HoldCycle::Step{HoldCycle::Wait::Drain, s, s, m,
                                           TileID{}, Port{}, Port{}, -1}});
            break;
          }
      }
      break;
    }
    case Kind::Anywhere:
    case Kind::Past: {
      llvm::SmallDenseSet<size_t, 8> behind;
      if (node.kind == Kind::Past)
        for (int h = node.index; h >= 0; h = u.parent[h])
          behind.insert(h);
      for (size_t h = 0; h < u.hops.size(); h++)
        if (!behind.contains(h))
          out.push_back({u.base + h, std::nullopt});
      for (size_t m = 0; m < u.members.size(); m++)
        out.push_back({receiverNode(node.tree, m), std::nullopt});
      break;
    }
    }
    return *(edges[n] = std::move(out));
  };

  // A cycle of receivers alone waiting on each other is the design's own, so
  // only cycles through a wait some requested routing adds count.
  auto counts = [&](const Edge &e) {
    if (!e.step || e.step->wait == HoldCycle::Wait::Drain)
      return false;
    return e.step->waiting < numRequested || e.step->sharer < numRequested ||
           e.step->holding < numRequested;
  };
  for (size_t n = 0; n < nodes.size(); n++)
    if (nodes[n].kind == Kind::Hop && llvm::any_of(successors(n), counts))
      roots.push_back(n);

  const WaitGraph graph{successors, nodes.size()};
  std::vector<int> component(nodes.size() + 1, -1);
  int components = 0;
  for (auto scc = llvm::scc_begin(&graph); !scc.isAtEnd(); ++scc, ++components)
    for (WaitNode n : *scc)
      component[n.second] = components;

  // A counted wait within one component lies on a closed walk. A packet holds
  // its arbiter until its tail passes, so no state has two trees holding one
  // arbiter, and a walk that needs that is no deadlock. The search fixes or
  // rules out a holder per arbiter until the shortest walk left agrees. Each
  // clash splits it in two, so past maxWalkSearches it keeps the first walk,
  // which at worst steers the router off a routing that cannot deadlock.
  using Grant = std::pair<TileID, int>;
  struct Holders {
    std::optional<size_t> fixed;
    SmallVector<size_t, 2> excluded;
  };
  using Constraints = std::map<Grant, Holders>;
  auto allowed = [](const Edge &edge, const Constraints &c) {
    if (!edge.step || edge.step->wait != HoldCycle::Wait::Arbiter)
      return true;
    auto it = c.find({edge.step->tile, edge.step->arbiter});
    if (it == c.end())
      return true;
    size_t holder = edge.step->holding;
    return it->second.fixed ? *it->second.fixed == holder
                            : !llvm::is_contained(it->second.excluded, holder);
  };
  auto closeWalk =
      [&](size_t x, const Edge &e,
          const Constraints &c) -> std::optional<SmallVector<const Edge *>> {
    if (!allowed(e, c))
      return std::nullopt;
    DenseMap<size_t, std::pair<size_t, const Edge *>> via;
    via[e.to] = {e.to, nullptr};
    std::deque<size_t> worklist{e.to};
    while (!via.contains(x)) {
      if (worklist.empty())
        return std::nullopt;
      size_t n = worklist.front();
      worklist.pop_front();
      for (const Edge &next : successors(n))
        if (component[next.to] == component[x] && allowed(next, c) &&
            via.try_emplace(next.to, n, &next).second)
          worklist.push_back(next.to);
    }
    SmallVector<const Edge *> path;
    for (size_t n = x; n != e.to; n = via.at(n).first)
      path.push_back(via.at(n).second);
    path.push_back(&e);
    std::reverse(path.begin(), path.end());
    return path;
  };
  auto toCycle = [](ArrayRef<const Edge *> path) {
    HoldCycle cycle;
    for (const Edge *edge : path)
      if (edge->step)
        cycle.steps.push_back(*edge->step);
    return cycle;
  };
  constexpr int maxWalkSearches = 1024;
  for (size_t x : roots)
    for (const Edge &e : successors(x)) {
      if (!counts(e) || component[e.to] != component[x])
        continue;
      std::optional<SmallVector<const Edge *>> first;
      SmallVector<Constraints> pending{{}};
      for (int search = 0; !pending.empty(); search++) {
        if (search == maxWalkSearches) {
          LLVM_DEBUG(llvm::dbgs() << "Hold cycle search gave up after "
                                  << search << " walks\n");
          return toCycle(*first);
        }
        Constraints c = pending.pop_back_val();
        std::optional<SmallVector<const Edge *>> path = closeWalk(x, e, c);
        if (!path)
          continue;
        if (!first)
          first = path;
        std::map<Grant, size_t> held;
        std::optional<std::pair<Grant, size_t>> clash;
        for (const Edge *edge : *path) {
          if (!edge->step || edge->step->wait != HoldCycle::Wait::Arbiter)
            continue;
          auto [it, inserted] = held.try_emplace(
              {edge->step->tile, edge->step->arbiter}, edge->step->holding);
          if (!inserted && it->second != edge->step->holding) {
            clash = *it;
            break;
          }
        }
        if (!clash) {
          LLVM_DEBUG(llvm::dbgs() << "Hold cycle search closed a walk after "
                                  << search + 1 << " walks\n");
          return toCycle(*path);
        }
        Constraints fix = c, exclude = c;
        fix[clash->first].fixed = clash->second;
        exclude[clash->first].excluded.push_back(clash->second);
        pending.push_back(std::move(exclude));
        pending.push_back(std::move(fix));
      }
    }
  return std::nullopt;
}

std::string StreamConflicts::explain(const HoldCycle &cycle) const {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (auto [i, step] : llvm::enumerate(cycle.steps)) {
    if (i)
      os << ' ';
    switch (step.wait) {
    case HoldCycle::Wait::Link:
      os << describeStream(streams[step.waiting]) << " can queue behind "
         << describeStream(streams[step.holding]) << " on "
         << describePort(step.sharerInput) << " into tile (" << step.tile.col
         << ", " << step.tile.row << ").";
      break;
    case HoldCycle::Wait::Arbiter:
      os << describeStream(streams[step.holding]) << " can hold arbiter "
         << step.arbiter << " at tile (" << step.tile.col << ", "
         << step.tile.row << ") that " << describeStream(streams[step.sharer])
         << " needs";
      if (step.waiting != step.sharer)
        os << ", and " << describeStream(streams[step.waiting])
           << " can queue behind it";
      os << ".";
      break;
    case HoldCycle::Wait::Drain:
      os << getAnalysis().explainBlock(step.waiting, step.holding);
      break;
    }
  }
  return s;
}
