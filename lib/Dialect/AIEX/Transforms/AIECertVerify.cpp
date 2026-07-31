//===- AIECertVerify.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECERTVERIFY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "cert-verify"

namespace {

// The uC (CERT group) id of a cert op is the group_id of its nearest enclosing
// cert.attach_to_group, or 0 if it is at device level.
static int getUcGroupId(Operation *op) {
  auto group = op->getParentOfType<AIEX::CertAttachToGroupOp>();
  return group ? group.getGroupID() : 0;
}

// G-localbar: within each uC scope, all cert.local_barrier ops that share a
// local_barrier_id must reside in the same cert.page (their participants must
// be co-resident on one page, else the firmware hangs).
static LogicalResult checkLocalBarrierColocation(AIE::DeviceOp device) {
  LogicalResult result = success();
  // Key: (uC group id, local_barrier_id) -> the co-location scope the id was
  // first seen in. Participants of a local barrier must share one page; a
  // barrier in a standalone job (no page) uses its enclosing job as its scope,
  // so barriers in *different* standalone jobs are correctly flagged as not
  // co-located (and two in the same job are fine). local_barrier always has an
  // enclosing job (HasParent<CertJobOp>), so the scope is never null.
  std::map<std::pair<int, int>, Operation *> firstScope;
  device.walk(
      [&](AIEX::CertLocalBarrierOp barrier) {
        int uc = getUcGroupId(barrier);
        int id = barrier.getLocalBarrierId();
        Operation *scope = barrier->getParentOfType<AIEX::CertPageOp>();
        if (!scope)
          scope = barrier->getParentOfType<AIEX::CertJobOp>();
        auto key = std::make_pair(uc, id);
        auto it = firstScope.find(key);
        if (it == firstScope.end()) {
          firstScope[key] = scope;
          return;
        }
        if (it->second != scope) {
          barrier.emitError()
              << "cert.local_barrier with local_barrier_id " << id << " in uC "
              << uc << " must be on the same page as its other participants";
          result = failure();
        }
      });
  return result;
}

// G-tct: within one page, at most one *job* may wait on a given (tile_id,
// channel_id). The firmware tracks a single outstanding TCT count per actor, so
// two jobs that are cooperatively scheduled on the same page and both wait on
// the same actor over-wait and hang forever.
//
// The constraint is on concurrency, not on the whole uC program, so two cases
// that look like duplicates are legal and must not be flagged:
//   - two wait_tcts in the *same job*: a job is a single thread of control, so
//     they execute in sequence and only one is ever outstanding;
//   - two wait_tcts on *different pages* of one uC: pages execute strictly in
//     order, so the earlier wait has retired before the later page starts.
// A page belongs to exactly one uC, so keying on the page scope also keys on
// the uC; a wait in a bare job (no enclosing page, e.g. before
// cert-legalize-pages has run) uses that job as its scope.
static LogicalResult checkWaitTctsUniqueness(AIE::DeviceOp device) {
  LogicalResult result = success();
  // (page scope, tile, channel) -> the job that already waits on that actor.
  std::map<std::tuple<Operation *, int, int>, Operation *> seen;
  device.walk([&](AIEX::CertWaitTCTSOp wait) {
    int tile = wait.getTileId();
    int channel = wait.getChannelId();
    Operation *job = wait->getParentOfType<AIEX::CertJobOp>();
    Operation *scope = wait->getParentOfType<AIEX::CertPageOp>();
    if (!scope)
      scope = job;
    auto key = std::make_tuple(scope, tile, channel);
    auto it = seen.find(key);
    if (it == seen.end()) {
      seen[key] = job;
      return;
    }
    // Same job: sequential within one thread of control, so legal.
    if (it->second == job)
      return;
    wait.emitError() << "more than one job on this cert.page waits on tile "
                     << tile << " channel " << channel << " in uC "
                     << getUcGroupId(wait)
                     << "; at most one job per page may wait_tcts on a given "
                        "(tile, channel)";
    result = failure();
  });
  return result;
}

// The set of uCs (CERT groups) present in a design: every group_id declared by
// a cert.attach_to_group, plus 0 when there is cert content at device level.
static std::set<int> computePresentUcs(AIE::DeviceOp device) {
  std::set<int> present;
  device.walk([&](AIEX::CertAttachToGroupOp group) {
    present.insert(group.getGroupID());
  });
  if (!device.getBody()->getOps<AIEX::CertPageOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertJobOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertSectionOp>().empty())
    present.insert(0);
  return present;
}

// G-preempt: every uC must expose the same multiset of cert.preempt ids. A uC
// is the group_id of an enclosing cert.attach_to_group, or 0 for device-level
// content. Compare each present uC's preempt-id multiset against the smallest
// present uC (the reference); a mismatch means a preemption point is missing or
// extra on some uC, which the firmware requires to be consistent.
static LogicalResult checkPreemptConsistency(AIE::DeviceOp device) {
  std::map<int, std::multiset<int>> preemptIds;
  std::map<int, AIEX::CertAttachToGroupOp> repAttach;
  std::set<int> present;

  // Every attach_to_group establishes a present uC.
  device.walk([&](AIEX::CertAttachToGroupOp group) {
    int uc = group.getGroupID();
    present.insert(uc);
    if (!repAttach.count(uc))
      repAttach[uc] = group;
  });

  // Device-level cert content lives on uC 0.
  if (!device.getBody()->getOps<AIEX::CertPageOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertJobOp>().empty() ||
      !device.getBody()->getOps<AIEX::CertSectionOp>().empty())
    present.insert(0);

  // Collect the preempt ids per uC.
  device.walk([&](AIEX::CertPreemptOp preempt) {
    int uc = getUcGroupId(preempt);
    present.insert(uc);
    preemptIds[uc].insert(preempt.getId());
  });

  if (present.size() < 2)
    return success();

  int ref = *present.begin();
  const std::multiset<int> &refSet = preemptIds[ref];
  LogicalResult result = success();
  for (int uc : present) {
    if (uc == ref)
      continue;
    if (preemptIds[uc] != refSet) {
      Operation *loc = repAttach.count(uc) ? repAttach[uc].getOperation()
                                           : device.getOperation();
      loc->emitError()
          << "cert.preempt ids on uC " << uc << " differ from uC " << ref
          << "; all uCs must have the same preempt ids (G-preempt)";
      result = failure();
    }
  }
  return result;
}

// Group/placement ids must be valid microcontroller indices for the device:
// nonnegative, less than the number of uCs, and (defensively) < 32 so they are
// representable in a party mask. Rejecting invalid ids up front keeps later
// checks (e.g. the remote-barrier mask shift) free of undefined behavior.
static LogicalResult checkGroupIds(AIE::DeviceOp device) {
  const AIE::AIETargetModel &tm = device.getTargetModel();
  int64_t numUcs =
      (int64_t)tm.columns() * (int64_t)tm.getNumMicrocontrollersPerColumn();
  LogicalResult result = success();
  auto checkId = [&](Operation *op, int64_t id, llvm::StringRef what) {
    if (id < 0 || id >= numUcs || id >= 32) {
      op->emitError() << what << " " << id << " is not a valid uC id [0, "
                      << numUcs << ")";
      result = failure();
    }
  };
  device.walk([&](AIEX::CertAttachToGroupOp g) {
    checkId(g, g.getGroupID(), "cert.attach_to_group group id");
  });
  // The verifier may run before placement lowering; check any leftover attrs.
  device.walk([&](AIEX::CertPageOp p) {
    if (auto pl = p.getPlacementAttr())
      checkId(p, pl.getInt(), "cert.page placement");
  });
  return result;
}

// G-uC / rendezvous: cross-uC remote barriers must actually rendezvous. Group
// occurrences by remote_barrier_id and require: all occurrences agree on
// party_mask; every mask bit names a present uC that has a matching occurrence
// (so nobody waits forever); each participating uC sets its own bit; and at
// most one occurrence per uC per id (one participant per column). Assumes group
// ids are already validated by checkGroupIds (shifts are still guarded).
static LogicalResult checkRemoteBarrierParties(AIE::DeviceOp device) {
  std::set<int> present = computePresentUcs(device);
  LogicalResult result = success();

  std::map<int, SmallVector<AIEX::CertRemoteBarrierOp, 4>> byId;
  device.walk([&](AIEX::CertRemoteBarrierOp b) {
    byId[b.getRemoteBarrierId()].push_back(b);
  });

  for (auto &entry : byId) {
    int id = entry.first;
    SmallVector<AIEX::CertRemoteBarrierOp, 4> &occs = entry.second;

    // All occurrences of an id must describe the same rendezvous.
    uint32_t mask = occs.front().getPartyMask();
    bool masksAgree = llvm::all_of(occs, [&](AIEX::CertRemoteBarrierOp b) {
      return b.getPartyMask() == mask;
    });
    if (!masksAgree) {
      for (auto b : occs)
        b.emitError() << "cert.remote_barrier id " << id
                      << " occurrences disagree on party_mask";
      result = failure();
      continue;
    }

    // Participating uCs (from occurrences); own-bit + one-per-uC checks.
    std::map<int, int> occPerUc;
    for (auto b : occs) {
      int own = getUcGroupId(b);
      occPerUc[own]++;
      if (own >= 0 && own < 32 && !((mask >> own) & 1u)) {
        b.emitError() << "cert.remote_barrier id " << id << " on uC " << own
                      << " excludes its own uC from party_mask";
        result = failure();
      }
    }
    for (auto &[uc, count] : occPerUc)
      if (count > 1) {
        occs.front().emitError()
            << "cert.remote_barrier id " << id << " has " << count
            << " occurrences on uC " << uc
            << " (at most one participant per uC per rendezvous)";
        result = failure();
      }

    // Every mask bit must be a present uC that has a matching barrier.
    for (int bit = 0; bit < 32; ++bit) {
      if (!((mask >> bit) & 1u))
        continue;
      if (!present.count(bit)) {
        occs.front().emitError()
            << "cert.remote_barrier id " << id << " party_mask references uC "
            << bit << " which is not present in the design";
        result = failure();
      } else if (!occPerUc.count(bit)) {
        occs.front().emitError()
            << "cert.remote_barrier id " << id << " party_mask includes uC "
            << bit << " but that uC has no matching remote_barrier(" << id
            << ") to rendezvous with";
        result = failure();
      }
    }
  }
  return result;
}

// G-barlimits: local_barrier_id must be in [0, 15] and remote_barrier_id in
// [1, 8]; a single uC may use at most 16 distinct local ids and 8 distinct
// remote ids (the firmware barrier-id budget per microcontroller).
static LogicalResult checkBarrierLimits(AIE::DeviceOp device) {
  LogicalResult result = success();
  std::map<int, std::set<int>> localIds;
  std::map<int, std::set<int>> remoteIds;

  device.walk([&](AIEX::CertLocalBarrierOp barrier) {
    int id = barrier.getLocalBarrierId();
    if (id < 0 || id > 15) {
      barrier.emitError() << "cert.local_barrier local_barrier_id " << id
                          << " out of range [0, 15]";
      result = failure();
    }
    localIds[getUcGroupId(barrier)].insert(id);
  });

  device.walk([&](AIEX::CertRemoteBarrierOp barrier) {
    int id = barrier.getRemoteBarrierId();
    if (id < 1 || id > 8) {
      barrier.emitError() << "cert.remote_barrier remote_barrier_id " << id
                          << " out of range [1, 8]";
      result = failure();
    }
    remoteIds[getUcGroupId(barrier)].insert(id);
  });

  for (auto &[uc, ids] : localIds)
    if (ids.size() > 16) {
      device.emitError() << "uC " << uc << " uses " << ids.size()
                         << " distinct local_barrier ids (max 16)";
      result = failure();
    }
  for (auto &[uc, ids] : remoteIds)
    if (ids.size() > 8) {
      device.emitError() << "uC " << uc << " uses " << ids.size()
                         << " distinct remote_barrier ids (max 8)";
      result = failure();
    }
  return result;
}

struct AIECertVerifyPass
    : xilinx::AIEX::impl::AIECertVerifyBase<AIECertVerifyPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    if (failed(checkGroupIds(device)))
      signalPassFailure();
    if (failed(checkLocalBarrierColocation(device)))
      signalPassFailure();
    if (failed(checkWaitTctsUniqueness(device)))
      signalPassFailure();
    if (failed(checkPreemptConsistency(device)))
      signalPassFailure();
    if (failed(checkRemoteBarrierParties(device)))
      signalPassFailure();
    if (failed(checkBarrierLimits(device)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECertVerifyPass() {
  return std::make_unique<AIECertVerifyPass>();
}
