//===- AIEShimSharing.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEShimSharing.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <tuple>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEANNOTATESHIMSHARING
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

ShimTransferSpans::ShimTransferSpans(
    DeviceOp device, function_ref<std::optional<End>(StringAttr)> endOf) {
  analyze(device, [&](DeviceOp owner, StringAttr symbol) {
    return owner == device ? endOf(symbol) : std::nullopt;
  });
}

ShimTransferSpans::ShimTransferSpans(
    ModuleOp module,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf) {
  analyze(module, endOf);
}

void ShimTransferSpans::analyze(
    Operation *root,
    function_ref<std::optional<End>(DeviceOp, StringAttr)> endOf) {
  struct Span {
    StringAttr endpoint;
    int64_t first, last;
  };
  struct Summary {
    SmallVector<Span> spans;
    int64_t duration = 0;
  };
  DenseMap<Operation *, Summary> summaries;
  DenseSet<Operation *> active;
  NamedOpTableCollection names;
  std::function<Summary(RuntimeSequenceOp)> summarize =
      [&](RuntimeSequenceOp sequence) -> Summary {
    if (active.contains(sequence)) {
      unknown = true;
      return {};
    }
    if (auto found = summaries.find(sequence); found != summaries.end())
      return found->second;
    active.insert(sequence);
    DeviceOp device = sequence->getParentOfType<DeviceOp>();
    Summary result;
    DenseMap<Operation *, int64_t> position, end;
    std::function<void(Operation *)> number = [&](Operation *op) {
      position[op] = result.duration++;
      if (auto run = dyn_cast<AIEX::RunOp>(op)) {
        if (auto callee = run.getCalleeRuntimeSequenceOp(&names))
          result.duration += summarize(callee).duration;
        else
          unknown = true;
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            number(&nested);
      end[op] = result.duration++;
    };
    number(sequence);
    auto interval = [&](Operation *op, int64_t from, int64_t to) {
      for (Operation *parent = op->getParentOp(); parent != sequence;
           parent = parent->getParentOp())
        if (isa<LoopLikeOpInterface>(parent)) {
          from = position[parent];
          to = end[parent];
        }
      return std::make_pair(from, to);
    };
    // Completion names a specific start, so a callee cannot retire caller work.
    auto note = [&](StringAttr endpoint, int64_t at,
                    std::optional<size_t> completion = std::nullopt) {
      if (completion) {
        result.spans[*completion].last = at;
        return *completion;
      }
      used.insert(endpoint);
      result.spans.push_back({endpoint, at, end[sequence]});
      return result.spans.size() - 1;
    };
    DenseMap<StringAttr, SmallVector<size_t>> pending;
    DenseMap<Value, StringAttr> tasks;
    DenseMap<size_t, Value> owners;
    auto transfer = [&](Operation *op, StringAttr symbol, bool starts,
                        bool all = false, Value owner = {}) {
      auto endpoint = endOf(device, symbol);
      if (!endpoint)
        return;
      auto [from, to] = interval(op, position[op], position[op]);
      auto &queue = pending[symbol];
      if (starts) {
        size_t start = note(endpoint->fifo, from);
        owners[start] = owner;
        queue.push_back(start);
      } else if (!endpoint->sends) {
        for (auto it = queue.begin(); it != queue.end();) {
          if (owner && owners.lookup(*it) != owner) {
            ++it;
            continue;
          }
          note(endpoint->fifo, to, *it);
          it = queue.erase(it);
          if (!all)
            break;
        }
      }
    };
    auto synchronous = [&](Operation *op, StringAttr endpoint, int64_t from,
                           int64_t to) {
      std::tie(from, to) = interval(op, from, to);
      auto start = note(endpoint, from);
      note(endpoint, to, start);
    };
    auto configuration = [&](Operation *op, DeviceOp configured,
                             std::optional<int> column = {}) {
      if (!configured)
        return;
      for (auto alloc : configured.getOps<ShimDMAAllocationOp>()) {
        if (!AIEDialect::IsCtrlPktOverlayAttrHelper(root->getContext())
                 .isAttrPresent(alloc) ||
            (column && alloc.getTileOp().colIndex() != *column))
          continue;
        if (auto endpoint = endOf(configured, alloc.getSymNameAttr()))
          synchronous(op, endpoint->fifo, position[op], position[op]);
      }
    };
    sequence.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto run = dyn_cast<AIEX::RunOp>(op)) {
        if (auto callee = run.getCalleeRuntimeSequenceOp(&names))
          for (const Span &span : summarize(callee).spans)
            synchronous(op, span.endpoint, position[op] + 1 + span.first,
                        position[op] + 1 + span.last);
      } else if (auto configure = dyn_cast<AIEX::ConfigureOp>(op)) {
        configuration(op, configure.getReferencedDeviceOp());
      } else if (isa<AIEX::NpuLoadPdiOp, AIEX::NpuWriteConfigOp,
                     AIEX::NpuWriteTeardownOp>(op)) {
        if (isa<AIEX::NpuLoadPdiOp>(op) && op->getParentOp() == sequence) {
          for (auto &[symbol, queue] : pending) {
            for (size_t start : queue)
              note(result.spans[start].endpoint, position[op] - 1, start);
            queue.clear();
          }
        }
        auto ref = op->getAttrOfType<FlatSymbolRefAttr>("device_ref");
        configuration(
            op, ref ? SymbolTable::lookupNearestSymbolFrom<DeviceOp>(op, ref)
                    : device);
      } else if (auto packet = dyn_cast<AIEX::NpuControlPacketOp>(op)) {
        configuration(op, device, packet.getColumnFromAddr());
      } else if (auto sync = dyn_cast<AIEX::NpuSyncOp>(op)) {
        SmallVector<int> values;
        for (Value operand : sync->getOperands()) {
          APInt value;
          if (!matchPattern(operand, m_ConstantInt(&value)))
            return;
          values.push_back(value.getSExtValue());
        }
        if (values[4] != 1 || values[5] != 1)
          return;
        StringAttr symbol;
        size_t first = std::numeric_limits<size_t>::max();
        for (const auto &[name, queue] : pending) {
          if (queue.empty() || queue.front() >= first)
            continue;
          Operation *owner = device.lookupSymbol(name);
          TileOp tile;
          std::optional<int> channel;
          DMAChannelDir direction;
          if (auto alloc = dyn_cast_or_null<ShimDMAAllocationOp>(owner)) {
            tile = alloc.getTileOp();
            channel = alloc.getChannelIndex();
            direction = alloc.getChannelDir();
          } else if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(owner)) {
            tile = endpoint.getTile().getDefiningOp<TileOp>();
            channel = endpoint.getChannelIndex();
            direction = endpoint.getRouteDirection();
          }
          if (tile && channel && tile.colIndex() == values[0] &&
              tile.rowIndex() == values[1] &&
              static_cast<int>(direction) == values[2] &&
              *channel == values[3]) {
            first = queue.front();
            symbol = name;
          }
        }
        if (symbol)
          transfer(op, symbol, false);
      } else if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
        transfer(op, memcpy.getMetadata().getRootReference(), true);
      } else if (auto wait = dyn_cast<AIEX::NpuDmaWaitOp>(op)) {
        transfer(op, wait.getSymbolAttr().getAttr(), false, true);
      } else if (auto configure = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
        tasks[configure.getResult()] = configure.getAlloc().getRootReference();
      } else if (isa<AIEX::DMAStartTaskOp, AIEX::DMAAwaitTaskOp,
                     AIEX::DMAFreeTaskOp>(op)) {
        if (auto symbol = tasks.lookup(op->getOperand(0)))
          transfer(op, symbol, isa<AIEX::DMAStartTaskOp>(op),
                   isa<AIEX::DMAFreeTaskOp>(op),
                   isa<AIEX::DMAAwaitTaskOp>(op) ? Value{} : op->getOperand(0));
      }
    });
    llvm::stable_sort(result.spans, [](const Span &a, const Span &b) {
      return a.first < b.first;
    });
    DenseMap<StringAttr, int64_t> live;
    for (const Span &span : result.spans) {
      lifetimes[span.endpoint].emplace_back(summaries.size(), span.first,
                                            span.last);
      for (const auto &[other, last] : live) {
        if (last < span.first)
          continue;
        overlaps.insert({span.endpoint, other});
        overlaps.insert({other, span.endpoint});
      }
      live[span.endpoint] = std::max(live.lookup(span.endpoint), span.last);
    }
    active.erase(sequence);
    summaries[sequence] = result;
    return result;
  };
  root->walk([&](Operation *op) {
    if (!isa<ObjectFifoCreateOp, RouteEndpointOp, ShimDMAAllocationOp>(op))
      return;
    DeviceOp device = op->getParentOfType<DeviceOp>();
    auto endpoint = endOf(device, SymbolTable::getSymbolName(op));
    if (!endpoint)
      return;
    bool sends = endpoint->sends;
    if (auto alloc = dyn_cast<ShimDMAAllocationOp>(op)) {
      sends = alloc.getChannelDir() == DMAChannelDir::MM2S;
      used.insert(endpoint->fifo);
    } else if (auto allowed =
                   AIEDialect::ShareableWithAttrHelper(op->getContext())
                       .getAttr(op)) {
      auto &peers = permissions[endpoint->fifo];
      for (Attribute attr : allowed)
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(attr))
          if (auto other = endOf(device, ref.getAttr()))
            peers.push_back(other->fifo);
    }
    if (auto fifo = dyn_cast<ObjectFifoCreateOp>(op))
      if (llvm::any_of(fifo.getConsumerTiles(), [](Value tile) {
            return cast<TileLike>(tile.getDefiningOp()).isShimTile();
          }))
        sends = false;
    // Receive ordering does not constrain when independent producers send.
    if (!sends)
      permissions.try_emplace(endpoint->fifo);
  });
  root->walk([&](RuntimeSequenceOp sequence) { summarize(sequence); });
}

bool ShimTransferSpans::isUsed(StringAttr endpoint) const {
  return used.contains(endpoint);
}

bool ShimTransferSpans::apart(StringAttr a, StringAttr b) const {
  auto permits = [&](StringAttr from, StringAttr to) {
    if (auto it = permissions.find(from); it != permissions.end())
      return llvm::is_contained(it->second, to);
    return !unknown && isUsed(from) && isUsed(to) &&
           !overlaps.contains({from, to});
  };
  return a != b && permits(a, b) && permits(b, a);
}

SmallVector<SmallVector<unsigned>>
ShimTransferSpans::groups(ArrayRef<StringAttr> ends) const {
  auto when = llvm::map_to_vector(
      ends, [&](StringAttr end) { return lifetimes.lookup(end); });
  auto order = llvm::to_vector(llvm::seq<unsigned>(0, ends.size()));
  llvm::stable_sort(order,
                    [&](unsigned x, unsigned y) { return when[x] < when[y]; });
  SmallVector<SmallVector<unsigned>> result;
  for (unsigned end : order) {
    auto *group = llvm::find_if(result, [&](ArrayRef<unsigned> members) {
      return llvm::all_of(
          members, [&](unsigned m) { return apart(ends[m], ends[end]); });
    });
    if (group == result.end())
      result.push_back({end});
    else
      group->push_back(end);
  }
  return result;
}

namespace {
struct AIEAnnotateShimSharingPass
    : xilinx::AIE::impl::AIEAnnotateShimSharingBase<
          AIEAnnotateShimSharingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    AIEDialect::IsCtrlPktOverlayAttrHelper overlay(&getContext());
    AIEDialect::ShareableWithAttrHelper sharing(&getContext());
    DenseMap<Operation *, StringAttr> keys;
    auto key = [&](Operation *op) {
      auto [it, inserted] = keys.try_emplace(op);
      if (inserted)
        it->second =
            StringAttr::get(&getContext(), std::to_string(keys.size()));
      return it->second;
    };
    ShimTransferSpans spans(
        module,
        [&](DeviceOp device,
            StringAttr symbol) -> std::optional<ShimTransferSpans::End> {
          Operation *op = device.lookupSymbol(symbol);
          if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(op)) {
            if (endpoint.getBundle() != WireBundle::DMA ||
                !endpoint.getTileLike().isShimTile())
              return std::nullopt;
            return ShimTransferSpans::End{key(endpoint),
                                          endpoint.getRouteDirection() ==
                                              DMAChannelDir::MM2S};
          }
          auto alloc = dyn_cast_or_null<ShimDMAAllocationOp>(op);
          if (!alloc)
            return std::nullopt;
          if (overlay.isAttrPresent(alloc)) {
            if (auto ref = device.getOverlayAttr())
              if (auto definition =
                      module.lookupSymbol<DeviceOp>(ref.getValue()))
                if (Operation *original = definition.lookupSymbol(symbol))
                  op = original;
            return ShimTransferSpans::End{key(op), false};
          }
          return ShimTransferSpans::End{key(op), alloc.getChannelDir() ==
                                                     DMAChannelDir::MM2S};
        });
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      auto annotate = [&](Operation *op, Value tile, DMAChannelDir direction,
                          StringAttr identity) {
        if (sharing.isAttrPresent(op))
          return;
        SmallVector<Attribute> allowed;
        for (auto other : device.getOps<RouteEndpointOp>())
          if (other != op && other.getBundle() == WireBundle::DMA &&
              other.getTile() == tile &&
              other.getRouteDirection() == direction &&
              spans.apart(identity, keys.lookup(other)))
            allowed.push_back(FlatSymbolRefAttr::get(other.getSymNameAttr()));
        sharing.setAttr(op, ArrayAttr::get(&getContext(), allowed));
      };
      for (auto endpoint : device.getOps<RouteEndpointOp>())
        if (endpoint.getBundle() == WireBundle::DMA &&
            endpoint.getTileLike().isShimTile())
          annotate(endpoint, endpoint.getTile(), endpoint.getRouteDirection(),
                   keys.lookup(endpoint));
      for (auto alloc : device.getOps<ShimDMAAllocationOp>())
        if (overlay.isAttrPresent(alloc)) {
          Operation *original = alloc;
          if (auto ref = device.getOverlayAttr())
            if (auto definition = module.lookupSymbol<DeviceOp>(ref.getValue()))
              if (Operation *source =
                      definition.lookupSymbol(alloc.getSymName()))
                original = source;
          annotate(alloc, alloc.getTile(), alloc.getChannelDir(),
                   keys.lookup(original));
        }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIEAnnotateShimSharingPass() {
  return std::make_unique<AIEAnnotateShimSharingPass>();
}
