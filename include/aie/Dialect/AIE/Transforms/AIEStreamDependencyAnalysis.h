//===- AIEStreamDependencyAnalysis.h ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace xilinx::AIE {

/// A tile port at the edge of the stream fabric.
struct StreamEndpoint {
  TileID tile;
  Port port;

  bool operator==(const StreamEndpoint &rhs) const {
    return std::tie(tile, port) == std::tie(rhs.tile, rhs.port);
  }
  bool operator!=(const StreamEndpoint &rhs) const { return !(*this == rhs); }
};

/// One DMA channel of one tile.
struct TileDMAChannel {
  TileID tile;
  DMAChannelDir dir;
  int channel;

  bool operator==(const TileDMAChannel &rhs) const {
    return std::tie(tile, dir, channel) ==
           std::tie(rhs.tile, rhs.dir, rhs.channel);
  }
  bool operator<(const TileDMAChannel &rhs) const {
    return std::tie(tile, dir, channel) <
           std::tie(rhs.tile, rhs.dir, rhs.channel);
  }
};

/// The BD chain a DMA channel runs, as one aie.dma_start, aie.dma or runtime
/// task programs it.
struct DmaChannelProgram {
  mlir::Operation *op;
  TileDMAChannel dma;
  llvm::SmallVector<mlir::Block *> bds{};
  llvm::SmallVector<mlir::Region *> regions{};
  /// Whether the BD chain runs forever.
  bool loops = false;
  /// The first of `bds` a looping chain returns to once it has run them all.
  size_t loopStart = 0;
  /// Times one start of the channel runs the chain: its repeat count plus one.
  uint64_t passes = 1;
};

/// A switchbox a stream passes: the port it enters by and, where it is packet
/// switched there, the arbiter it takes.
struct StreamHop {
  TileID tile;
  Port input;
  std::optional<int> arbiter;
};

/// One stream from a source tile port to a destination tile port, recovered
/// from the physical switchbox and shim-mux configuration.
struct RoutedStream {
  StreamEndpoint src;
  StreamEndpoint dst;
  /// The packet id carried, if any.
  std::optional<int> packetID;
  /// The destination stores each packet's header along with its payload.
  bool keepsPktHeader = false;
  /// The switchboxes the stream passes, source first, where it is routed.
  llvm::SmallVector<StreamHop, 8> hops{};
  /// The stream carries every id that agrees with `packetID` on these bits.
  int packetMask = ~0;
};

/// A cycle of waits packet streams can deadlock in, given their routes. A
/// packet holds the arbiter of every switchbox it has entered until its last
/// word leaves, so one stuck anywhere holds up whatever needs those arbiters,
/// or queues behind it on a link.
struct HoldCycle {
  enum class Wait {
    /// `waiting` queues behind `holding` on the port both enter `tile` by.
    Link,
    /// `waiting`, or `sharer` it queues behind, needs the arbiter `holding`
    /// holds at `tile`; `sharer` enters by `sharerInput` and `holding` by
    /// `holderInput`.
    Arbiter,
    /// `waiting` fills its receiver, and draining that waits on `holding`.
    Drain,
  };
  struct Step {
    Wait wait;
    size_t waiting;
    size_t sharer;
    size_t holding;
    TileID tile;
    Port sharerInput;
    Port holderInput;
    int arbiter;
    /// The wait holds however the arbiters are planned, as `sharer` and
    /// `holding` go into a receiver they share at `tile`, or have one id and
    /// go on as one from there.
    bool forced = false;
  };
  llvm::SmallVector<Step, 4> steps;
};

/// Follows every stream from its source tile port through the configured
/// switchboxes, using tile geometry for the links between them. A packet
/// stream is traced once per id its source sends, as read off the BDs and
/// runtime transfers that program the source; with none known, once per
/// rule of the first switchbox it enters.
std::vector<RoutedStream> traceRoutedStreams(DeviceOp device);

/// The streams the aie.flow and aie.packet_flow ops ask for, one per source,
/// destination and packet id, before any of them is routed.
std::vector<RoutedStream> requestedStreams(DeviceOp device);

/// Names the stream by its endpoints and packet id, e.g.
/// "packet flow (0, 1) DMA:0 -> (0, 2) DMA:1 (id 3)".
std::string describeStream(const RoutedStream &stream);

/// How much data a stream carries over a run and how much a receiver takes in
/// before it has to wait on another agent, read off BD lengths, repeat
/// counts, lock initial values and runtime transfers.
class StreamVolumeAnalysis {
public:
  StreamVolumeAnalysis(DeviceOp device, llvm::ArrayRef<RoutedStream> streams);
  StreamVolumeAnalysis(const StreamVolumeAnalysis &) = delete;
  StreamVolumeAnalysis &operator=(const StreamVolumeAnalysis &) = delete;

  /// Bytes the stream's source sends with the stream's packet id, headers
  /// included where the receiver keeps them, or nullopt when that is
  /// unbounded or unknown. A BD chain that loops sends only what the lock
  /// tokens other agents can ever release to it let it.
  std::optional<uint64_t> sendVolume(const RoutedStream &stream) const;

  /// Bytes the receiver at `endpoint` accepts before it waits on another
  /// agent, or nullopt when it never does. Where the analysis gives up before
  /// the receiver waits, the bytes it took in so far.
  std::optional<uint64_t> receiveCapacity(const StreamEndpoint &endpoint) const;

  /// Whether `streams` can send the receiver at `endpoint` more than it takes
  /// in before it waits on another agent.
  bool canFill(const StreamEndpoint &endpoint,
               llvm::ArrayRef<RoutedStream> streams) const;

  /// Whether the source both streams come from can send a packet of `then`
  /// after one of `first`; nullopt if its program doesn't show or they come
  /// from different sources.
  std::optional<bool> maySendAfter(const RoutedStream &first,
                                   const RoutedStream &then) const;

private:
  /// Bytes the looping BD chain `program` sends before an acquire runs out of
  /// tokens, counting each BD as `bytesOf` says.
  std::optional<uint64_t>
  loopedVolume(const DmaChannelProgram &program,
               llvm::function_ref<uint64_t(DMABDOp)> bytesOf) const;
  /// Visits, in order, the BDs the looping BD chain `program` runs until an
  /// acquire runs out of tokens. False if it cannot tell when that is.
  bool walkLoop(const DmaChannelProgram &program,
                llvm::function_ref<void(DMABDOp)> visit) const;
  /// Tokens every agent but `self` can release to `lock` over a run. A core
  /// runs its body once.
  std::optional<uint64_t> tokensFromOthers(LockOp lock,
                                           const DmaChannelProgram *self) const;
  std::optional<uint64_t> releasesOver(const DmaChannelProgram &program,
                                       LockOp lock) const;

  mutable DeviceOp device;
  llvm::ArrayRef<RoutedStream> streams;
  std::map<TileDMAChannel, llvm::SmallVector<DmaChannelProgram, 1>> programs;
  /// The npu.dma_memcpy_nd ops that run on each shim channel.
  std::map<TileDMAChannel, llvm::SmallVector<mlir::Operation *>> memcpys;
  /// The channel program each use_lock in a BD chain belongs to.
  llvm::DenseMap<mlir::Operation *, const DmaChannelProgram *> lockUseProgram;
  mutable llvm::DenseSet<const DmaChannelProgram *> visiting;
};

/// Which agent waits on which. An agent is a core, one DMA channel, or the
/// controller of a tile, which sends the task-complete tokens the host waits
/// for on its TileControl port. P waits on Q when P acquires a lock Q
/// releases, when P sends a stream Q receives or the reverse, or when P is a
/// runtime-issued channel the host issues only after waiting on Q. Waiting on
/// a channel waits on the controllers of its column too, whose tokens tell
/// the host it is done. A channel with no program in the design may wait on
/// anything on its tile. A receiving channel that takes in all it is sent
/// before its locks run out waits on no lock. Program order within an agent
/// is not modeled.
class StreamWaitGraph {
public:
  /// A core, a DMA channel, or a tile's controller. A core's or controller's
  /// `dma` names only its tile.
  struct Agent {
    enum class Kind { Channel, Core, Controller };
    TileDMAChannel dma;
    Kind kind = Kind::Channel;
    bool onShim = false;

    static Agent core(TileID tile) {
      return {{tile, DMAChannelDir::S2MM, 0}, Kind::Core};
    }
    static Agent controller(TileID tile) {
      return {{tile, DMAChannelDir::S2MM, 0}, Kind::Controller};
    }
    static Agent channel(const TileDMAChannel &dma) { return {dma}; }
    /// Each agent at a stream endpoint: the core, DMA channel or controller
    /// pushing data into (`sending`) or pulling data out of the fabric there.
    static std::optional<Agent> at(const StreamEndpoint &endpoint,
                                   bool sending);
  };

  StreamWaitGraph(DeviceOp device, llvm::ArrayRef<RoutedStream> streams,
                  const StreamVolumeAnalysis &volumes);

  /// The agent pushing data into (`sending`) or pulling data out of the
  /// fabric at `endpoint`, if the endpoint has one.
  std::optional<unsigned> agentAt(const StreamEndpoint &endpoint,
                                  bool sending) const;

  /// Agents whose progress frees room in `agent` for more incoming data: a
  /// core frees itself, a receiving channel waits on whoever releases the
  /// locks it acquires and, if the host issues it, on what the host waits for
  /// first.
  llvm::SmallVector<unsigned> drainersOf(unsigned agent) const;

  /// Whether any agent in `targets` is reachable from `from` without passing
  /// through `avoid`.
  bool reaches(llvm::ArrayRef<unsigned> from, llvm::ArrayRef<unsigned> targets,
               llvm::ArrayRef<unsigned> avoid) const;

  /// A shortest chain of waits from an agent in `from` to one in `targets`
  /// that avoids `avoid`, both ends included; empty when there is none.
  llvm::SmallVector<unsigned> waitChain(llvm::ArrayRef<unsigned> from,
                                        llvm::ArrayRef<unsigned> targets,
                                        llvm::ArrayRef<unsigned> avoid) const;

  /// Whether the design programs this agent, so its waits are known rather
  /// than assumed.
  bool isModeled(unsigned id) const { return modeled.contains(id); }

  const Agent &getAgent(unsigned id) const { return agents[id]; }
  std::string describe(unsigned id) const;

private:
  enum class EdgeKind { Lock, Stream, Host };
  struct Edge {
    unsigned to;
    EdgeKind kind;
  };

  unsigned getOrCreate(const Agent &agent);
  std::optional<unsigned> lookup(const Agent &agent) const;
  void addEdge(unsigned from, unsigned to, EdgeKind kind);

  std::vector<Agent> agents;
  std::vector<llvm::SmallVector<Edge, 4>> edges;
  std::map<std::pair<Agent::Kind, TileDMAChannel>, unsigned> agentIDs;
  llvm::DenseSet<unsigned> modeled;
};

/// Which streams can deadlock against each other if they share an arbiter or
/// a link. A packet holds its arbiter grant until tlast, so a stream stalled
/// at a full receiver holds up whatever else waits on that grant.
class StreamDeadlockAnalysis {
public:
  /// `streams` must outlive the analysis.
  StreamDeadlockAnalysis(DeviceOp device, llvm::ArrayRef<RoutedStream> streams);

  /// Whether stream `f`, stalled at its receiver, can keep stream `g` from
  /// ever arriving: `f` carries more than its receiver takes in before
  /// waiting, and draining that receiver waits on `g`.
  bool canBlock(size_t f, size_t g) const;

  /// Whether stream `f` carries nothing, so it neither holds nor waits.
  bool silent(size_t f) const;

  /// Why `f` can block `g`: the chain of waits from `f`'s receiver to `g`, and
  /// which links of it are assumed for lack of information. Requires
  /// canBlock(f, g).
  std::string explainBlock(size_t f, size_t g) const;

  /// What explainBlock(f, g) takes on trust rather than reads off the design,
  /// one sentence each. Requires canBlock(f, g).
  llvm::SmallVector<std::string> assumptions(size_t f, size_t g) const;

private:
  bool canStall(size_t f) const;
  llvm::SmallVector<unsigned> blockingChain(size_t f, size_t g) const;

  llvm::ArrayRef<RoutedStream> streams;
  StreamVolumeAnalysis volumes;
  StreamWaitGraph graph;
  /// canStall, silent and canBlock, once asked.
  mutable std::vector<std::optional<bool>> stalls, silence;
  mutable llvm::DenseMap<std::pair<size_t, size_t>, bool> blocks;
};

/// The streams a device asks for or already routes, and which pairs of them
/// must not share an arbiter or a link. The analysis runs on the first query.
class StreamConflicts {
public:
  explicit StreamConflicts(DeviceOp device);
  StreamConflicts(const StreamConflicts &) = delete;
  StreamConflicts &operator=(const StreamConflicts &) = delete;

  llvm::ArrayRef<RoutedStream> getStreams() const { return streams; }

  /// The streams the flow ops ask for, which come first in getStreams().
  llvm::ArrayRef<RoutedStream> getRequestedStreams() const {
    return llvm::ArrayRef(streams).take_front(numRequested);
  }

  /// Whether `s` and `t` can deadlock if they share an arbiter or a link.
  /// Streams from one source are serialized there anyway, and streams into
  /// one destination already wait on each other there, so neither conflicts.
  /// The packets one source sends with one id move down every branch as one,
  /// so the same holds for any two streams of such trees.
  bool conflict(size_t s, size_t t) const;

  /// Whether a routing that puts packet streams `s` and `t` on one arbiter
  /// anywhere has a hold cycle: they conflict, or the tree of one, stuck at a
  /// receiver, waits on the other through trees that each block the next.
  bool mustSeparate(size_t s, size_t t) const;

  /// Why `s` and `t` conflict. Requires mustSeparate(s, t), or (s, t) from
  /// unavoidable().
  std::string explain(size_t s, size_t t) const;

  /// The requested streams where the first can hold up the second however
  /// they are routed, as one source or receiver already orders them. Only
  /// pairs with a packet stream count: circuits take no arbiter, so a wait
  /// between two of them is the design's, not a hazard of any routing. Nor do
  /// pairs that need StreamDeadlockAnalysis::assumptions: the routing cannot
  /// help those either, and the design may well not do what is assumed.
  llvm::SmallVector<std::pair<size_t, size_t>> unavoidable() const;

  /// A cycle of waits the packet streams can deadlock in when routed along
  /// `routes`, indexed like getStreams(), that the routing of some requested
  /// stream takes part in; nullopt when there is none. Waits between trees
  /// at the master port of a receiver they share hold however they are
  /// routed, so they count only with `forcedWaits`. With `definite`, drains
  /// that need StreamDeadlockAnalysis::assumptions do not count.
  std::optional<HoldCycle>
  holdCycle(llvm::ArrayRef<llvm::SmallVector<StreamHop, 8>> routes,
            bool forcedWaits = false, bool definite = false) const;

  /// The waits of `cycle`, one sentence each.
  std::string explain(const HoldCycle &cycle) const;

private:
  bool blocks(size_t s, size_t t) const;
  bool related(size_t s, size_t t) const;
  /// What makes a packet stream one with the others of its tree: the same
  /// source, the same id, and whether a flow op asks for it.
  using TreeKey = std::tuple<TileID, Port, int, bool>;
  /// The tree stream `s` belongs to; nullopt for a circuit stream.
  std::optional<TreeKey> treeKey(size_t s) const;
  /// The packet trees tree `a` waits on through trees that each block the
  /// next, each with the pair of streams, the first of the tree before it,
  /// that it was reached by.
  const llvm::DenseMap<size_t, std::pair<size_t, size_t>> &
  waitsFrom(size_t a) const;
  const StreamDeadlockAnalysis &getAnalysis() const;

  DeviceOp device;
  std::vector<RoutedStream> streams;
  size_t numRequested;
  /// The streams of each tree, and the tree of each stream.
  std::vector<llvm::SmallVector<size_t, 2>> treeMembers;
  std::vector<size_t> treeOf;
  mutable std::vector<
      std::optional<llvm::DenseMap<size_t, std::pair<size_t, size_t>>>>
      waits;
  mutable std::optional<StreamDeadlockAnalysis> analysis;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H
