//===- AIECheckAgainstImage.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"

#include <map>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECHECKAGAINSTIMAGE
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

using Coords = std::pair<int, int>;

/// Where a DMA channel is: its tile, direction and index.
struct Channel {
  Coords tile;
  DMAChannelDir dir;
  int index;
  bool operator<(const Channel &other) const {
    return std::tie(tile, dir, index) <
           std::tie(other.tile, other.dir, other.index);
  }
  bool operator==(const Channel &other) const {
    return !(*this < other) && !(other < *this);
  }
};

struct Buffer {
  Coords tile;
  std::optional<int64_t> address, bank;
};

struct Lock {
  Coords tile;
  std::optional<int64_t> id;
};

struct Chain {
  Channel channel;
  SmallVector<int64_t> bdIds;
};

/// The static allocation of one device, each resource under a name that does
/// not depend on where it was put: a shim end under its allocation's symbol,
/// a buffer or lock under its own, a DMA chain or runtime task under the
/// buffers it moves, a switchbox under its tile.
struct Allocation {
  std::map<std::string, Channel> shimEnds;
  std::map<std::string, Buffer> buffers;
  std::map<std::string, Lock> locks;
  std::map<std::string, Chain> chains;
  std::map<std::string, std::set<Channel>> tasks;
  std::map<std::pair<Coords, std::string>, std::set<std::string>> switches;
};

Coords coordsOf(Value tile) {
  auto op = cast<TileOp>(tile.getDefiningOp());
  return {op.colIndex(), op.rowIndex()};
}

std::string nameOf(Operation *op) {
  if (auto name =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return name.str();
  return "";
}

/// The named buffers, each with the offset it is moved at, or failing them the
/// named locks, that `ops` touch: a name for a DMA chain or task that the
/// inputs of one join, writing one buffer at different offsets, do not share.
std::string identity(ArrayRef<Operation *> ops) {
  std::set<std::string> buffers, locks;
  for (Operation *op : ops) {
    if (auto bd = dyn_cast<DMABDOp>(op)) {
      if (auto access = bd.getBuffer().getDefiningOp<GetBufferOp>())
        if (auto buffer = access.getBufferOp())
          if (std::string name = nameOf(buffer); !name.empty())
            buffers.insert(name + "@" +
                           std::to_string(bd.getConstantOffset().value_or(-1)));
    } else if (auto use = dyn_cast<UseLockOp>(op)) {
      if (auto lock = use.getLockOp())
        if (std::string name = nameOf(lock); !name.empty())
          locks.insert(name);
    }
  }
  return llvm::join(buffers.empty() ? locks : buffers, ", ");
}

Allocation allocationOf(DeviceOp device) {
  Allocation allocation;
  for (auto alloc : device.getOps<ShimDMAAllocationOp>())
    allocation.shimEnds[alloc.getSymName().str()] = {
        coordsOf(alloc.getTile()), alloc.getChannelDir(),
        static_cast<int>(alloc.getChannelIndex())};
  for (auto buffer : device.getOps<BufferOp>())
    if (std::string name = nameOf(buffer); !name.empty())
      allocation.buffers[name] = {coordsOf(buffer.getTile()),
                                  buffer.getAddress(), buffer.getMemBank()};
  for (auto lock : device.getOps<LockOp>())
    if (std::string name = nameOf(lock); !name.empty())
      allocation.locks[name] = {coordsOf(lock.getTile()), lock.getLockID()};

  for (auto program : device.getOps<DmaBody>())
    for (Block &block : program.getDmaBody())
      for (auto start : block.getOps<DMAStartOp>()) {
        SmallVector<Operation *> ops;
        SmallVector<int64_t> bdIds;
        SmallPtrSet<Block *, 8> seen;
        for (Block *at = start.getDest(); at && seen.insert(at).second;) {
          if (at->getOps<DMABDOp>().empty())
            break;
          for (Operation &op : *at) {
            ops.push_back(&op);
            if (auto bd = dyn_cast<DMABDOp>(op))
              bdIds.push_back(bd.getBdId().value_or(-1));
          }
          auto next = dyn_cast<NextBDOp>(at->getTerminator());
          at = next ? next.getDest() : nullptr;
        }
        std::string name = identity(ops);
        if (name.empty())
          continue;
        allocation.chains[name] = {{coordsOf(program.getTile()),
                                    start.getChannelDir(),
                                    start.getChannelIndex()},
                                   bdIds};
      }

  device.walk([&](AIEX::DMAConfigureTaskOp task) {
    SmallVector<Operation *> ops;
    task.getBody().walk([&](Operation *op) { ops.push_back(op); });
    if (std::string name = identity(ops); !name.empty())
      allocation.tasks[name].insert({coordsOf(task.getTile()),
                                     task.getDirection(),
                                     static_cast<int>(task.getChannel())});
  });

  auto addSwitch = [&](Operation *box, Value tile, StringRef kind) {
    auto &connects = allocation.switches[{coordsOf(tile), kind.str()}];
    box->walk([&](ConnectOp connect) {
      connects.insert((stringifyWireBundle(connect.getSourceBundle()) + ":" +
                       Twine(connect.sourceIndex()) + "->" +
                       stringifyWireBundle(connect.getDestBundle()) + ":" +
                       Twine(connect.destIndex()))
                          .str());
    });
  };
  for (auto box : device.getOps<SwitchboxOp>())
    addSwitch(box, box.getTile(), "switchbox");
  for (auto mux : device.getOps<ShimMuxOp>())
    addSwitch(mux, mux.getTile(), "shim mux");
  return allocation;
}

std::string str(Coords tile) {
  return ("(" + Twine(tile.first) + ", " + Twine(tile.second) + ")").str();
}

std::string str(const Channel &channel) {
  return (str(channel.tile) + " " + stringifyDMAChannelDir(channel.dir) + " " +
          Twine(channel.index))
      .str();
}

std::string str(const std::set<Channel> &channels) {
  SmallVector<std::string> parts;
  for (const Channel &channel : channels)
    parts.push_back(str(channel));
  return llvm::join(parts, " and ");
}

std::string str(std::optional<int64_t> value) {
  return value ? std::to_string(*value) : "none";
}

/// `names`, a long list shortened to its first two and a count.
std::string shortList(ArrayRef<std::string> names) {
  if (names.size() <= 3)
    return llvm::join(names, ", ");
  return names[0] + ", " + names[1] + " and " +
         std::to_string(names.size() - 2) + " more";
}

/// An identity's first two parts and a count, for display.
std::string shortName(StringRef name) {
  SmallVector<StringRef> parts;
  name.split(parts, ", ");
  return shortList(llvm::to_vector(
      llvm::map_range(parts, [](StringRef part) { return part.str(); })));
}

/// What differs between the image's allocation and this stream's, each line
/// a resource and what to pin.
SmallVector<std::string> differences(const Allocation &image,
                                     const Allocation &stream) {
  SmallVector<std::string> lines;
  auto both = [](const auto &a, const auto &b, auto fn) {
    std::set<std::string> keys;
    for (auto &[key, _] : a)
      keys.insert(key);
    for (auto &[key, _] : b)
      keys.insert(key);
    for (const std::string &key : keys) {
      auto ia = a.find(key), ib = b.find(key);
      fn(key, ia == a.end() ? nullptr : &ia->second,
         ib == b.end() ? nullptr : &ib->second);
    }
  };

  both(image.shimEnds, stream.shimEnds,
       [&](const std::string &name, const Channel *i, const Channel *s) {
         StringRef fifo = StringRef(name).rsplit("_shim_alloc").first;
         if (i && s && *i == *s)
           return;
         lines.push_back((fifo + "'s shim end: image " +
                          (i ? str(*i) : "none") + ", this stream " +
                          (s ? str(*s) : "none") +
                          "; pin it to keep it stable across shapes")
                             .str());
       });

  // A tile that moved shows in every buffer it holds; name it once.
  std::map<std::pair<Coords, Coords>, SmallVector<std::string>> moved;
  for (auto &[name, i] : image.buffers)
    if (auto it = stream.buffers.find(name);
        it != stream.buffers.end() && it->second.tile != i.tile)
      moved[{i.tile, it->second.tile}].push_back(name);
  for (auto &[tiles, names] : moved)
    lines.push_back("the tile holding " + shortList(names) + ": image " +
                    str(tiles.first) + ", this stream " + str(tiles.second) +
                    "; pin it to keep it stable across shapes");

  both(image.chains, stream.chains,
       [&](const std::string &name, const Chain *i, const Chain *s) {
         if (i && s && i->channel == s->channel) {
           auto ids = [](ArrayRef<int64_t> bdIds) {
             return llvm::join(
                 llvm::map_range(bdIds,
                                 [](int64_t id) { return std::to_string(id); }),
                 ", ");
           };
           if (i->bdIds != s->bdIds)
             lines.push_back("the BD ids of the chain moving " +
                             shortName(name) + " at " + str(i->channel) +
                             ": image " + ids(i->bdIds) + ", this stream " +
                             ids(s->bdIds));
           return;
         }
         lines.push_back("the DMA channel moving " + shortName(name) +
                         ": image " + (i ? str(i->channel) : "none") +
                         ", this stream " + (s ? str(s->channel) : "none") +
                         "; pin its endpoint to keep it stable across shapes");
       });

  both(image.tasks, stream.tasks,
       [&](const std::string &name, const std::set<Channel> *i,
           const std::set<Channel> *s) {
         // A task only one sequence issues is no allocation of its own.
         if (!i || !s || *i == *s)
           return;
         lines.push_back(
             "the runtime tasks moving " + shortName(name) + ": image " +
             str(*i) + ", this stream " + str(*s) +
             "; pin their endpoint to keep it stable across shapes");
       });

  both(image.buffers, stream.buffers,
       [&](const std::string &name, const Buffer *i, const Buffer *s) {
         if (i && s && i->tile != s->tile)
           return;
         if (i && s && i->address == s->address && i->bank == s->bank)
           return;
         if (!i || !s) {
           lines.push_back("buffer " + name + ": only " +
                           (i ? "the image" : "this stream") + " has it, at " +
                           str(i ? i->tile : s->tile));
           return;
         }
         lines.push_back("buffer " + name + " on " + str(i->tile) +
                         ": image address " + str(i->address) + " (bank " +
                         str(i->bank) + "), this stream " + str(s->address) +
                         " (bank " + str(s->bank) + ")");
       });

  both(image.locks, stream.locks,
       [&](const std::string &name, const Lock *i, const Lock *s) {
         if (i && s && i->tile == s->tile && i->id == s->id)
           return;
         if (i && s && i->tile != s->tile)
           return;
         if (!i || !s) {
           lines.push_back("lock " + name + ": only " +
                           (i ? "the image" : "this stream") + " has it");
           return;
         }
         lines.push_back("lock " + name + " on " + str(i->tile) +
                         ": image id " + str(i->id) + ", this stream id " +
                         str(s->id));
       });

  size_t routes = 0;
  std::set<std::pair<Coords, std::string>> boxes;
  for (auto &[key, _] : image.switches)
    boxes.insert(key);
  for (auto &[key, _] : stream.switches)
    boxes.insert(key);
  for (auto &key : boxes) {
    std::set<std::string> none;
    auto ia = image.switches.find(key), is = stream.switches.find(key);
    const auto &i = ia == image.switches.end() ? none : ia->second;
    const auto &s = is == stream.switches.end() ? none : is->second;
    SmallVector<std::string> onlyImage, onlyStream;
    std::set_difference(i.begin(), i.end(), s.begin(), s.end(),
                        std::back_inserter(onlyImage));
    std::set_difference(s.begin(), s.end(), i.begin(), i.end(),
                        std::back_inserter(onlyStream));
    if (onlyImage.empty() && onlyStream.empty())
      continue;
    ++routes;
    lines.push_back(
        key.second + " " + str(key.first) + ": image connects " +
        (onlyImage.empty() ? "none" : llvm::join(onlyImage, ", ")) +
        ", this stream " +
        (onlyStream.empty() ? "none" : llvm::join(onlyStream, ", ")));
  }
  if (routes)
    lines.push_back("the routes follow the placement and channels above");
  return lines;
}

struct AIECheckAgainstImagePass
    : xilinx::AIE::impl::AIECheckAgainstImageBase<AIECheckAgainstImagePass> {
  using AIECheckAgainstImageBase::AIECheckAgainstImageBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (clImage.empty()) {
      module.emitError("aie-check-against-image needs image=<module>");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> image =
        parseSourceFile<ModuleOp>(clImage, ParserConfig(&getContext()));
    if (!image) {
      module.emitError("could not read the image's module ") << clImage;
      return signalPassFailure();
    }
    DenseMap<StringAttr, DeviceOp> imageDevices;
    for (auto device : image->getOps<DeviceOp>())
      imageDevices[device.getSymNameAttr()] = device;

    bool mismatch = false;
    for (auto stream : module.getOps<DeviceOp>()) {
      DeviceOp target = imageDevices.lookup(stream.getSymNameAttr());
      if (!target) {
        stream.emitOpError("has no device of its name in the image's module ")
            << clImage;
        mismatch = true;
        continue;
      }
      SmallVector<std::string> lines =
          differences(allocationOf(target), allocationOf(stream));
      if (lines.empty())
        continue;
      auto diag = mlir::emitError(stream.getLoc())
                  << "this instruction stream would program resources its "
                     "image ("
                  << clImage << ") does not have; " << lines.size()
                  << " differ:";
      for (const std::string &line : lines)
        diag << "\n  " << line;
      mismatch = true;
    }
    if (mismatch)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECheckAgainstImagePass() {
  return std::make_unique<AIECheckAgainstImagePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECheckAgainstImagePass(
    const AIECheckAgainstImageOptions &options) {
  return std::make_unique<AIECheckAgainstImagePass>(options);
}
