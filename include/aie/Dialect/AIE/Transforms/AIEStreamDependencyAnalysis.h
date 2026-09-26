//===- AIEStreamDependencyAnalysis.h ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

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
  /// The packet id carried, or nullopt for a circuit-switched stream.
  std::optional<int> packetID;
  /// The destination stores each packet's header along with its payload.
  bool keepsPktHeader = false;
  /// The switchboxes the stream passes, source first, where it is routed.
  llvm::SmallVector<StreamHop, 8> hops;
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
  explicit StreamVolumeAnalysis(DeviceOp device);

  /// Bytes the stream's source sends with the stream's packet id, headers
  /// included where the receiver keeps them, or nullopt when that is
  /// unbounded or unknown.
  std::optional<uint64_t> sendVolume(const RoutedStream &stream) const;

  /// Bytes the receiver at `endpoint` accepts before it waits on another
  /// agent, or nullopt when it never does.
  std::optional<uint64_t> receiveCapacity(const StreamEndpoint &endpoint) const;

private:
  mutable DeviceOp device;
  std::map<std::tuple<int, int, DMAChannelDir, int>,
           llvm::SmallVector<mlir::Operation *, 2>>
      programs;
};

/// Which agent waits on which. An agent is a core or one DMA channel. P waits
/// on Q when P acquires a lock Q releases, when P sends a stream Q receives or
/// the reverse, or when P is a runtime-issued channel the host issues only
/// after waiting on Q. A channel with no program in the design may wait on
/// anything on its tile. Program order within an agent is not modeled.
class StreamWaitGraph {
public:
  struct Agent {
    TileID tile;
    bool isCore;
    DMAChannelDir dir;
    int channel;
  };

  StreamWaitGraph(DeviceOp device, llvm::ArrayRef<RoutedStream> streams);

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

  unsigned getOrCreate(TileID tile, bool isCore, DMAChannelDir dir,
                       int channel);
  std::optional<unsigned> lookup(TileID tile, bool isCore, DMAChannelDir dir,
                                 int channel) const;
  void addEdge(unsigned from, unsigned to, EdgeKind kind);

  std::vector<Agent> agents;
  std::vector<llvm::SmallVector<Edge, 4>> edges;
  std::map<std::tuple<int, int, bool, int, int>, unsigned> agentIDs;
  llvm::DenseSet<unsigned> modeled;
};

/// Which streams can deadlock against each other if they share an arbiter or
/// a link. A packet holds its arbiter grant until tlast, so a stream stalled
/// at a full receiver holds up whatever else waits on that grant.
class StreamDeadlockAnalysis {
public:
  StreamDeadlockAnalysis(DeviceOp device, std::vector<RoutedStream> streams);

  /// Whether stream `f`, stalled at its receiver, can keep stream `g` from
  /// ever arriving: `f` carries more than its receiver takes in before
  /// waiting, and draining that receiver waits on `g`.
  bool canBlock(size_t f, size_t g) const;

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

  std::vector<RoutedStream> streams;
  StreamVolumeAnalysis volumes;
  StreamWaitGraph graph;
  mutable std::map<size_t, bool> stalls;
  mutable std::map<std::pair<size_t, size_t>, bool> blocks;
};

/// The streams a device asks for or already routes, and which pairs of them
/// must not share an arbiter or a link. The analysis runs on the first query.
class StreamConflicts {
public:
  explicit StreamConflicts(DeviceOp device);

  llvm::ArrayRef<RoutedStream> getStreams() const { return streams; }

  /// The streams the flow ops ask for, which come first in getStreams().
  llvm::ArrayRef<RoutedStream> getRequestedStreams() const {
    return llvm::ArrayRef(streams).take_front(numRequested);
  }

  /// Whether `s` and `t` can deadlock if they share an arbiter or a link.
  /// Streams from one source are serialized there anyway, and streams into
  /// one destination already wait on each other there, so neither conflicts.
  bool conflict(size_t s, size_t t);

  /// Why `s` and `t` conflict. Requires conflict(s, t).
  std::string explain(size_t s, size_t t);

  /// A cycle of waits the packet streams can deadlock in when routed along
  /// `routes`, indexed like getStreams(), that the routing of some requested
  /// stream takes part in; nullopt when there is none.
  std::optional<HoldCycle>
  holdCycle(llvm::ArrayRef<llvm::SmallVector<StreamHop, 8>> routes);

  /// The waits of `cycle`, one sentence each.
  std::string explain(const HoldCycle &cycle);

private:
  bool blocks(size_t s, size_t t);

  DeviceOp device;
  std::vector<RoutedStream> streams;
  size_t numRequested;
  std::optional<StreamDeadlockAnalysis> analysis;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESTREAMDEPENDENCYANALYSIS_H
