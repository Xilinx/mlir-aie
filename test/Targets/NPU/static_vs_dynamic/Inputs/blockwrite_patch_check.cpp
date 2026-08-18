//===- blockwrite_patch_check.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stand-in for the aiebu ELF-packager invariant we cannot run in this
// environment: every DDR_PATCH (address_patch) opcode must patch a register
// address that lies inside the [addr, addr + 4*num_payload_words) range of a
// BLOCKWRITE opcode that appears EARLIER in the stream. aiebu rejects a stream
// that violates this with "No block-write opcode present before the patch
// opcode for address 0x...". Plain XRT dispatch does not enforce it, so the
// dynamic BD-pool lowering could regress silently; this checker catches that
// from the pure TXN word stream, no aiebu required.
//
// The generated builder (which defines GEN_FN) is #included via -DGEN_HDR;
// GEN_FN / ARGVAL come from -D so this one file serves any dynamic sequence.
//
// It also carries a permanent NEGATIVE self-test: an inline hand-built stream
// with a patch NOT covered by any preceding block-write, asserted to be
// REJECTED by the same checker. A checker that always passes is worse than no
// checker, so we prove it can fail before we trust it to pass.
//
//===----------------------------------------------------------------------===//

#include GEN_HDR

#include "aie/Runtime/TxnEncoding.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

namespace {

using namespace aie_runtime;

// One parsed instruction, enough to check the invariant and to print a listing.
struct Op {
  uint32_t opcode;
  size_t wordPos;    // index of the opcode word in the stream
  uint64_t addr;     // block-write base / patched register (0 if N/A)
  size_t payloadLen; // block-write payload word count (0 if N/A)
};

// A block-write's covered register range, tagged with its position in the
// stream so we can enforce ORDER, not just membership.
struct Range {
  uint64_t lo;    // inclusive
  uint64_t hi;    // exclusive: addr + 4*payloadLen
  size_t wordPos; // where the block-write appeared
};

// Parse a TXN word stream into a flat op list. Returns false (and prints) on a
// truncated stream or an unknown opcode -- either would make the invariant
// check unsound, so we refuse rather than skip. Layout indices match
// TxnEncoding.h exactly (txn_append_*); the size-in-bytes word is at a
// different index per opcode, so each case reads it explicitly.
bool parseStream(const char *name, const std::vector<uint32_t> &txn,
                 std::vector<Op> &ops) {
  if (txn.size() < 4) {
    std::fprintf(stderr, "%s: stream too short for 4-word header\n", name);
    return false;
  }
  size_t pos = 4; // skip the 4-word header
  auto have = [&](size_t need) -> bool {
    if (pos + need <= txn.size())
      return true;
    std::fprintf(stderr,
                 "%s: truncated stream: opcode 0x%x at word %zu needs %zu "
                 "words, only %zu remain\n",
                 name, txn[pos], pos, need, txn.size() - pos);
    return false;
  };
  while (pos < txn.size()) {
    uint32_t opc = txn[pos];
    switch (opc) {
    case TXN_OPC_WRITE: {
      if (!have(6))
        return false;
      ops.push_back({opc, pos, txn[pos + 2], 0});
      pos += 6;
      break;
    }
    case TXN_OPC_BLOCKWRITE: {
      // Layout: [op][col/row][addr][byteSize=(4+count)*4][payload...].
      if (!have(4))
        return false;
      uint32_t addr = txn[pos + 2];
      uint32_t byteSize = txn[pos + 3];
      if (byteSize % sizeof(uint32_t) != 0) {
        std::fprintf(stderr, "%s: blockwrite byteSize %u not word-aligned\n",
                     name, byteSize);
        return false;
      }
      size_t total = byteSize / sizeof(uint32_t);
      if (total < 4) {
        std::fprintf(stderr,
                     "%s: blockwrite byteSize %u smaller than 4-word header\n",
                     name, byteSize);
        return false;
      }
      if (!have(total))
        return false;
      ops.push_back({opc, pos, addr, total - 4});
      pos += total;
      break;
    }
    case TXN_OPC_MASKWRITE: {
      if (!have(7))
        return false;
      ops.push_back({opc, pos, txn[pos + 2], 0});
      pos += 7;
      break;
    }
    case TXN_OPC_TCT: { // sync: [op][byteSize=16][.][.]
      if (!have(4))
        return false;
      ops.push_back({opc, pos, 0, 0});
      pos += 4;
      break;
    }
    case TXN_OPC_DDR_PATCH: {
      // Layout: [op][byteSize=48][.][.][.][action][addr][.][argIdx]...
      if (!have(12))
        return false;
      ops.push_back({opc, pos, txn[pos + 6], 0});
      pos += 12;
      break;
    }
    default:
      std::fprintf(stderr, "%s: unhandled TXN opcode 0x%x at word %zu\n", name,
                   opc, pos);
      return false;
    }
  }
  return true;
}

const char *opName(uint32_t opc) {
  switch (opc) {
  case TXN_OPC_WRITE:
    return "WRITE";
  case TXN_OPC_BLOCKWRITE:
    return "BLOCKWRITE";
  case TXN_OPC_MASKWRITE:
    return "MASKWRITE";
  case TXN_OPC_TCT:
    return "SYNC";
  case TXN_OPC_DDR_PATCH:
    return "DDR_PATCH";
  default:
    return "?";
  }
}

void printListing(const char *name, const std::vector<Op> &ops) {
  std::printf("--- parsed op listing for %s ---\n", name);
  for (const Op &o : ops) {
    if (o.opcode == TXN_OPC_BLOCKWRITE)
      std::printf("  word %4zu  %-10s addr=0x%08llx  payload_words=%zu  covers "
                  "[0x%08llx, 0x%08llx)\n",
                  o.wordPos, opName(o.opcode), (unsigned long long)o.addr,
                  o.payloadLen, (unsigned long long)o.addr,
                  (unsigned long long)(o.addr + 4 * o.payloadLen));
    else if (o.opcode == TXN_OPC_DDR_PATCH || o.opcode == TXN_OPC_WRITE ||
             o.opcode == TXN_OPC_MASKWRITE)
      std::printf("  word %4zu  %-10s addr=0x%08llx\n", o.wordPos,
                  opName(o.opcode), (unsigned long long)o.addr);
    else
      std::printf("  word %4zu  %-10s\n", o.wordPos, opName(o.opcode));
  }
}

// The invariant. Returns true iff every DDR_PATCH is covered by a BLOCKWRITE
// that appears strictly earlier in the stream. `verbose` prints each matched
// patch so a PASS is inspectable, not just a silent exit 0.
bool checkBlockwriteCoversPatch(const char *name, const std::vector<Op> &ops,
                                bool verbose) {
  std::vector<Range> ranges;
  bool ok = true;
  int patchCount = 0;
  for (const Op &o : ops) {
    if (o.opcode == TXN_OPC_BLOCKWRITE) {
      ranges.push_back({o.addr, o.addr + 4 * o.payloadLen, o.wordPos});
      continue;
    }
    if (o.opcode != TXN_OPC_DDR_PATCH)
      continue;
    ++patchCount;
    // A range qualifies only if it both covers the address AND appeared before
    // this patch in stream order. `ranges` only ever holds block-writes seen so
    // far, so every entry already satisfies the order constraint -- but we
    // assert wordPos < patch.wordPos explicitly so the property is checked, not
    // merely implied by traversal order.
    const Range *cover = nullptr;
    for (const Range &r : ranges) {
      if (r.wordPos < o.wordPos && o.addr >= r.lo && o.addr < r.hi) {
        cover = &r;
        break;
      }
    }
    if (!cover) {
      std::fprintf(stderr,
                   "%s: VIOLATION: DDR_PATCH at word %zu patches address "
                   "0x%08llx with no preceding block-write covering it "
                   "(aiebu would reject: \"No block-write opcode present "
                   "before the patch opcode for address 0x%llx\")\n",
                   name, o.wordPos, (unsigned long long)o.addr,
                   (unsigned long long)o.addr);
      ok = false;
      continue;
    }
    if (verbose)
      std::printf("  OK: DDR_PATCH at word %zu addr=0x%08llx covered by "
                  "BLOCKWRITE at word %zu [0x%08llx, 0x%08llx)\n",
                  o.wordPos, (unsigned long long)o.addr, cover->wordPos,
                  (unsigned long long)cover->lo, (unsigned long long)cover->hi);
  }
  if (ok && verbose)
    std::printf("%s: all %d DDR_PATCH op(s) covered by a preceding "
                "block-write\n",
                name, patchCount);
  return ok;
}

// PERMANENT NEGATIVE SELF-TEST. Hand-build a stream with a DDR_PATCH whose
// address is NOT covered by any preceding block-write and assert the checker
// REJECTS it. If this ever "passes" (checker returns true), the checker is
// vacuous and the whole test is worthless -- so a false-negative here fails the
// program. Also exercise the order constraint: a block-write that COVERS the
// address but appears AFTER the patch must still be rejected.
bool runNegativeSelfTest() {
  bool allGood = true;

  // Case A: patch with zero block-writes anywhere.
  {
    std::vector<uint32_t> t;
    txn_init(t);
    txn_append_address_patch(t, /*addr=*/0x1D000, /*arg_idx=*/0,
                             /*arg_plus=*/0);
    txn_prepend_header(t, 1u);
    std::vector<Op> ops;
    if (!parseStream("negA", t, ops)) {
      std::fprintf(stderr, "negA: parse failed (self-test broken)\n");
      allGood = false;
    } else if (checkBlockwriteCoversPatch("negA", ops, /*verbose=*/false)) {
      std::fprintf(stderr,
                   "SELF-TEST FAILURE: checker ACCEPTED a patch with no "
                   "block-write at all (negA); checker is vacuous\n");
      allGood = false;
    } else {
      std::printf("self-test negA: correctly rejected (patch, no "
                  "block-write)\n");
    }
  }

  // Case B: a block-write covers a DIFFERENT address than the patch targets.
  {
    std::vector<uint32_t> t;
    txn_init(t);
    uint32_t payload[2] = {0xAAAAAAAA, 0xBBBBBBBB};
    txn_append_blockwrite(t, /*addr=*/0x2000, payload, /*count=*/2);
    // Covered range is [0x2000, 0x2008). Patch 0x9000 is outside it.
    txn_append_address_patch(t, /*addr=*/0x9000, 0, 0);
    txn_prepend_header(t, 2u);
    std::vector<Op> ops;
    if (!parseStream("negB", t, ops)) {
      std::fprintf(stderr, "negB: parse failed (self-test broken)\n");
      allGood = false;
    } else if (checkBlockwriteCoversPatch("negB", ops, /*verbose=*/false)) {
      std::fprintf(stderr,
                   "SELF-TEST FAILURE: checker ACCEPTED a patch outside every "
                   "block-write range (negB); checker is vacuous\n");
      allGood = false;
    } else {
      std::printf("self-test negB: correctly rejected (patch outside "
                  "block-write range)\n");
    }
  }

  // Case C: the ONLY covering block-write appears AFTER the patch. This tests
  // the ORDER property specifically: membership alone would (wrongly) accept.
  {
    std::vector<uint32_t> t;
    txn_init(t);
    txn_append_address_patch(t, /*addr=*/0x3004, 0, 0);
    uint32_t payload[4] = {0, 0, 0, 0};
    txn_append_blockwrite(t, /*addr=*/0x3000, payload, /*count=*/4);
    txn_prepend_header(t, 2u);
    std::vector<Op> ops;
    if (!parseStream("negC", t, ops)) {
      std::fprintf(stderr, "negC: parse failed (self-test broken)\n");
      allGood = false;
    } else if (checkBlockwriteCoversPatch("negC", ops, /*verbose=*/false)) {
      std::fprintf(stderr,
                   "SELF-TEST FAILURE: checker ACCEPTED a patch whose covering "
                   "block-write comes LATER in the stream (negC); order "
                   "constraint not enforced\n");
      allGood = false;
    } else {
      std::printf("self-test negC: correctly rejected (covering block-write "
                  "after patch)\n");
    }
  }

  // Positive control: a block-write that covers the patched address and
  // precedes it MUST be accepted -- otherwise the checker rejects everything
  // and its PASS on real streams is meaningless.
  {
    std::vector<uint32_t> t;
    txn_init(t);
    uint32_t payload[4] = {0, 0, 0, 0};
    txn_append_blockwrite(t, /*addr=*/0x4000, payload, /*count=*/4);
    txn_append_address_patch(t, /*addr=*/0x4004, 0,
                             0); // inside [0x4000,0x4010)
    txn_prepend_header(t, 2u);
    std::vector<Op> ops;
    if (!parseStream("posCtl", t, ops)) {
      std::fprintf(stderr, "posCtl: parse failed (self-test broken)\n");
      allGood = false;
    } else if (!checkBlockwriteCoversPatch("posCtl", ops, /*verbose=*/false)) {
      std::fprintf(stderr,
                   "SELF-TEST FAILURE: checker REJECTED a validly-covered "
                   "patch (posCtl); checker rejects everything\n");
      allGood = false;
    } else {
      std::printf("self-test posCtl: correctly accepted (patch inside "
                  "preceding block-write)\n");
    }
  }

  return allGood;
}

} // namespace

int main() {
  // 1. Prove the checker actually detects violations before trusting a PASS.
  if (!runNegativeSelfTest()) {
    std::fprintf(stderr, "negative self-test failed; checker is unreliable\n");
    return 2;
  }

  // 2. Run the real dynamic BD-pool builder and check its stream.
  auto genOpt = GEN_FN(ARGVAL);
  if (!genOpt) {
    std::fprintf(stderr, "builder %s returned nullopt for arg=%d\n", "GEN_FN",
                 (int)(ARGVAL));
    return 1;
  }
  const std::vector<uint32_t> &txn = *genOpt;

  std::vector<Op> ops;
  if (!parseStream("generated", txn, ops))
    return 1;
  printListing("generated", ops);

  if (!checkBlockwriteCoversPatch("generated", ops, /*verbose=*/true)) {
    std::fprintf(stderr,
                 "generated stream violates the block-write-covers-patch "
                 "invariant\n");
    return 1;
  }

  std::printf("PASS: block-write-covers-patch invariant holds (%zu words, "
              "arg=%d)\n",
              txn.size(), (int)(ARGVAL));
  return 0;
}
