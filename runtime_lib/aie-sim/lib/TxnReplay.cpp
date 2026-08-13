//===- TxnReplay.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The TXN container format, version 0.1, and the operations mlir-aie emits.
//
// Provenance is the struct declarations the writer casts onto the buffer --
// XAie_TxnHeader and the XAie_*Hdr family, third_party/aie-rt/.../xaiegbl.h:
// 523-575 -- with opcode ids from xaie_txn.h:30-47. Field offsets are those
// structs' natural layout, and the two that carry padding are called out where
// they are read. Cross-checked against the parser mlir-aie already has,
// lib/Conversion/AIEToConfiguration/AIEToConfiguration.cpp:110-300, which is
// the same format read for a different purpose.
//
// Validated against the corpus rather than assumed: every sequence in the IRON
// build tree parses to exactly the operation count its header declares and
// consumes exactly its declared size, the largest being decode_fused's
// main_sequence.bin at 227376 operations over 14550592 bytes.
//
//===----------------------------------------------------------------------===//

#include "aiesim/TxnReplay.h"
#include "aiesim/Components.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// xaie_txn.h:30-47.
constexpr uint8_t kOpWrite = 0;
constexpr uint8_t kOpBlockWrite = 1;
constexpr uint8_t kOpMaskWrite = 3;
constexpr uint8_t kOpMaskPoll = 4;
constexpr uint8_t kOpPreempt = 6;   // Not in the enum; AIEToConfiguration:130.
constexpr uint8_t kOpLoadPdi = 8;   // Ditto, :131.
constexpr uint8_t kOpTct = 128;     // XAIE_IO_CUSTOM_OP_BEGIN.
constexpr uint8_t kOpDdrPatch = 129;

// XAie_TxnHeader, xaiegbl.h:523-533. Major/Minor/DevGen/NumRows/NumCols/
// NumMemTileRows are one byte each, then two of padding, then two words.
constexpr size_t kHeaderBytes = 16;
constexpr uint8_t kMajor = 0, kMinor = 1;

std::string atByte(size_t pos) {
  return " (at byte " + std::to_string(pos) + ")";
}

struct Reader {
  const std::vector<uint8_t> &d;

  bool have(size_t pos, size_t n) const { return pos + n <= d.size(); }
  uint32_t u32(size_t pos) const {
    uint32_t v;
    std::memcpy(&v, d.data() + pos, sizeof(v));
    return v;
  }
  uint64_t u64(size_t pos) const {
    uint64_t v;
    std::memcpy(&v, d.data() + pos, sizeof(v));
    return v;
  }
};

// Every operation begins with XAie_OpHdr {Op, Col, Row} and every one this
// format uses carries its own byte length, but NOT at a shared offset: the
// custom ops put it in XAie_CustomOpHdr's word at +4, while the core ops put it
// after their own fields. Returning 0 means "unknown opcode", which stops the
// replay rather than guessing a stride.
size_t operationSize(const Reader &r, size_t pos, uint8_t op) {
  switch (op) {
  case kOpWrite: // XAie_Write32Hdr: OpHdr, pad, RegOff(8), Value, Size.
    return r.have(pos, 24) ? r.u32(pos + 20) : 0;
  case kOpBlockWrite: // XAie_BlockWrite32Hdr: OpHdr, Col, Row, pad, RegOff,
                      // Size, then the payload.
    return r.have(pos, 16) ? r.u32(pos + 12) : 0;
  case kOpMaskWrite: // XAie_MaskWrite32Hdr / XAie_MaskPoll32Hdr: OpHdr, pad,
  case kOpMaskPoll:  // RegOff(8), Value, Mask, Size.
    return r.have(pos, 28) ? r.u32(pos + 24) : 0;
  case kOpPreempt: // TxnPreemptHeader, AIEToConfiguration.cpp:96-100.
    return 4;
  case kOpLoadPdi: // TxnLoadPdiHeader, :102-107.
    return 16;
  case kOpTct:
  case kOpDdrPatch: // XAie_CustomOpHdr {OpHdr, Size}.
    return r.have(pos, 8) ? r.u32(pos + 4) : 0;
  default:
    return 0;
  }
}

} // namespace

TxnOutcome aiesim::replayTxn(Array &array, const void *image, size_t sizeBytes,
                             uint64_t base, TxnReplayStats &stats,
                             std::string &error, uint64_t waitCycles) {
  std::vector<uint8_t> d(sizeBytes);
  if (sizeBytes)
    std::memcpy(d.data(), image, sizeBytes);
  Reader r{d};

  if (d.size() < kHeaderBytes) {
    error = "TXN image is shorter than its own header";
    return TxnOutcome::Malformed;
  }
  if (d[0] != kMajor || d[1] != kMinor) {
    char buf[112];
    std::snprintf(buf, sizeof(buf),
                  "TXN header version is %u.%u; only %u.%u is modelled (the "
                  "'optimized' layout packs different fields)",
                  d[0], d[1], kMajor, kMinor);
    error = buf;
    return TxnOutcome::Malformed;
  }
  const uint32_t declaredOps = r.u32(8);
  const uint32_t declaredSize = r.u32(12);
  if (declaredSize != d.size()) {
    char buf[112];
    std::snprintf(buf, sizeof(buf),
                  "TXN header declares %u bytes but the file holds %zu",
                  declaredSize, d.size());
    error = buf;
    return TxnOutcome::Malformed;
  }

  // Walk once before applying anything, to find out whether this sequence
  // loads its own configurations.
  //
  // A fused build emits two KINDS of sequence: one per fused op, each driving
  // the single context its own CDO group brings up, and one orchestrator that
  // loads each of those partitions in turn and drives them. Replaying the
  // orchestrator against one loaded configuration would program buffer
  // descriptors belonging to eighteen designs that are not there, then wait on
  // every one of them -- decode_fused's is 227376 operations with 6804 waits,
  // so the cost of finding that out by running it is the whole budget times
  // 6804. Declining is not a limitation of the parser: the operations parse
  // fine, and it is the RUN they describe that this model is not performing.
  //
  // Before the walk, so nothing is half applied and the caller's array is still
  // exactly what its CDO made it.
  uint32_t pdis = 0;
  for (size_t scan = kHeaderBytes; scan < d.size();) {
    size_t size = operationSize(r, scan, d[scan]);
    if (size == 0)
      break; // Let the real walk below produce the diagnostic.
    if (d[scan] == kOpLoadPdi)
      ++pdis;
    scan += size;
  }
  if (pdis != 0) {
    stats.loadPdi = pdis;
    char buf[176];
    std::snprintf(buf, sizeof(buf),
                  "TXN sequence loads %u partition image(s) of its own; it "
                  "orchestrates several configurations and this replay drives "
                  "the one already loaded, so none of it was applied",
                  pdis);
    error = buf;
    return TxnOutcome::Declined;
  }

  uint32_t applied = 0;
  // Task-completion tokens this sequence has already spent, keyed by
  // (col, row, direction, channel). Lives for the whole replay because a token
  // outlives the wait that could have taken it -- see the kOpTct case.
  std::map<uint64_t, uint32_t> tctConsumed;
  size_t pos = kHeaderBytes;
  while (pos < d.size()) {
    const uint8_t op = d[pos];
    const size_t size = operationSize(r, pos, op);
    if (size == 0) {
      char buf[144];
      std::snprintf(buf, sizeof(buf),
                    "unknown TXN opcode %u; refusing to skip it, because an "
                    "operation carries its own length and there is no way to "
                    "find the next one without it",
                    op);
      error = buf + atByte(pos);
      return TxnOutcome::Malformed;
    }
    if (!r.have(pos, size)) {
      char buf[112];
      std::snprintf(buf, sizeof(buf),
                    "TXN operation %u declares %zu bytes but only %zu remain",
                    op, size, d.size() - pos);
      error = buf + atByte(pos);
      return TxnOutcome::Malformed;
    }

    switch (op) {
    case kOpWrite: {
      uint64_t addr = base + (r.u32(pos + 8) | (uint64_t(r.u32(pos + 12)) << 32));
      array.write32(addr, r.u32(pos + 16));
      ++stats.write32;
      break;
    }
    case kOpMaskWrite: {
      uint64_t addr = base + (r.u32(pos + 8) | (uint64_t(r.u32(pos + 12)) << 32));
      uint32_t value = r.u32(pos + 16), mask = r.u32(pos + 20);
      array.write32(addr, (array.read32(addr) & ~mask) | (value & mask));
      ++stats.maskWrite32;
      break;
    }
    case kOpBlockWrite: {
      // RegOff here is a 32-bit field, not the 64-bit one the scalar ops
      // carry: XAie_BlockWrite32Hdr spends the space on its own Col/Row
      // instead. The column and row are already inside RegOff.
      uint64_t addr = base + r.u32(pos + 8);
      uint32_t words = static_cast<uint32_t>(size - 16) / 4;
      for (uint32_t i = 0; i < words; ++i)
        array.write32(addr + 4ull * i, r.u32(pos + 16 + 4 * i));
      ++stats.blockWrite32;
      stats.blockWriteWords += words;
      break;
    }
    case kOpMaskPoll: {
      uint64_t addr = base + (r.u32(pos + 8) | (uint64_t(r.u32(pos + 12)) << 32));
      uint32_t value = r.u32(pos + 16), mask = r.u32(pos + 20);
      uint64_t deadline = array.cycle() + waitCycles;
      // Array::read32 advances the clock, so this loop is itself what lets the
      // polled-for state arrive -- the same mechanism that makes an aie-rt host
      // wait work against this model.
      while ((array.read32(addr) & mask) != value && array.cycle() < deadline) {
      }
      if ((array.read32(addr) & mask) != value)
        ++stats.maskPollTimedOut;
      ++stats.maskPoll;
      break;
    }
    case kOpDdrPatch: {
      // patch_op_t (xaiegbl.h:593-599) after the 8-byte XAie_CustomOpHdr:
      // op_base b at +8 (two words), u32 action at +16, four bytes of padding
      // to align the u64s, then regaddr at +24, argidx at +32, argplus at +40.
      //
      // mlir-aie's parser reads `action` at +20 instead, which is that padding
      // (AIEToConfiguration.cpp:273). Both read 0 on every one of the 16900
      // patches in the corpus, so the disagreement is latent rather than
      // observable; this file follows the struct, and checks the other word
      // too so a future non-zero action cannot slip past whichever is right.
      if (!r.have(pos, 48)) {
        error = "TXN address patch is shorter than patch_op_t" + atByte(pos);
        return TxnOutcome::Malformed;
      }
      uint32_t action = r.u32(pos + 16), pad = r.u32(pos + 20);
      if (action != 0 || pad != 0) {
        char buf[144];
        std::snprintf(buf, sizeof(buf),
                      "TXN address patch has a non-zero action (0x%X at +16, "
                      "0x%X at +20); only 0 is modelled, and the two offsets "
                      "disagree on which word holds it",
                      action, pad);
        error = buf + atByte(pos);
        return TxnOutcome::Malformed;
      }
      uint64_t addr = base + r.u64(pos + 24);
      uint64_t argIdx = r.u64(pos + 32);
      uint64_t argPlus = r.u64(pos + 40);
      uint64_t value = argumentBase(static_cast<uint32_t>(argIdx)) + argPlus;

      // REPLACE, not add. The op's own field documentation is what decides it:
      // argplus is "value to add to what's passed @ argidx" (xaiegbl.h:598),
      // i.e. the addition is argument-plus-offset, and the result is the
      // address word. Read the other way -- add the argument base to whatever
      // the register holds -- argplus would have nothing to do.
      //
      // The two readings are not distinguishable by result on the designs
      // here, and the counter below is what says so instead of a comment
      // claiming it. Every block write that precedes a patch PRE-LOADS the
      // address word with exactly the argplus that follows it, so
      // `base + argplus` and `register + base` compute the same value: 1468 of
      // 1468 patches across all nineteen single-context sequences. The only
      // mismatches are inside decode_fused's main_sequence.bin, whose 312
      // load-pdi operations swap the configuration under us -- see below for
      // why this model does not follow those, which also makes its register
      // state there the wrong thing to compare against.
      if (array.read32(addr) != static_cast<uint32_t>(argPlus))
        ++stats.addressPatchPreloadMismatch;

      // `addr` names the BD's ADDRESS-LOW word. The high half is the next word
      // -- for a shim BD, AddrHigh's 16 bits share it with the packet fields
      // (Dma.cpp's makeShimLayout), so it is a masked write, not a store.
      //
      // Whether argplus is signed is NOT resolved: the struct declares u64 and
      // mlir-aie reads it as int32, the corpus fits both (every value is under
      // 2^32; read as int32 they span -1118077504..309766720), and no consumer
      // is in any tree here -- the firmware resolves these. It does not change
      // what this measures, because one argument per 4 GiB means either reading
      // lands inside that argument's own range and nowhere near another's.
      // It WOULD change which bytes of a buffer a design touched, so a data
      // check needs this answered first.
      array.write32(addr, static_cast<uint32_t>(value));
      constexpr uint32_t kAddrHighMask = 0x0000FFFFu;
      uint32_t high = static_cast<uint32_t>(value >> 32) & kAddrHighMask;
      array.write32(addr + 4,
                    (array.read32(addr + 4) & ~kAddrHighMask) | high);
      ++stats.addressPatch;
      break;
    }
    case kOpTct: {
      // The wait that makes a sequence blocking. The descriptor packs the tile
      // and direction, the config the channel and how many tiles to wait for
      // (AIEToConfiguration.cpp:242-251).
      uint32_t descriptor = r.u32(pos + 8), config = r.u32(pos + 12);
      uint32_t col = (descriptor >> 16) & 0xff;
      uint32_t row = (descriptor >> 8) & 0xff;
      auto dir = (descriptor & 0xff) == 0 ? DmaDirection::S2MM
                                          : DmaDirection::MM2S;
      uint32_t channel = (config >> 24) & 0xff;
      uint32_t colCount = (config >> 16) & 0xff;
      uint32_t rowCount = (config >> 8) & 0xff;

      // What the model can observe is a BD COMPLETING on that channel, not a
      // token being emitted: DmaModule counts completions and does not record
      // whether the completing BD had EnToken set. Every wait in the corpus
      // follows a queue push that set it (0x80000001 -- bit 31 is qEnToken),
      // so the two coincide there; a design that waited on a channel whose BDs
      // did not enable tokens would hang on hardware and would not here.
      //
      // A token is EMITTED by a completing BD and CONSUMED by a wait, so what a
      // wait needs is a completion this sequence has not already spent -- which
      // is why `tctConsumed` is carried across the whole replay rather than the
      // count being resampled per wait. Testing "did the counter change while I
      // waited" instead is EDGE-triggered, and wrong in exactly the case that
      // matters: op2_ElementwiseAdd pushes one BD per shim channel and issues
      // eight waits, and by the time the last four ran, their BDs had completed
      // during the earlier waits' spinning. Those four then waited for a SECOND
      // completion that was never coming, while the data they were waiting for
      // sat finished in DDR -- reported as a data-path stall for a design whose
      // data path had moved all 24 of its BDs and every byte, both directions.
      struct Target {
        DmaModule *dma;
        uint64_t key;
      };
      std::vector<Target> targets;
      for (uint32_t c = 0; c < colCount; ++c) {
        for (uint32_t w = 0; w < rowCount; ++w) {
          Tile *t = array.tile(col + c, row + w);
          if (!t || !t->dma())
            continue;
          uint64_t key = (static_cast<uint64_t>(col + c) << 24) |
                         (static_cast<uint64_t>(row + w) << 16) |
                         (static_cast<uint64_t>(dir == DmaDirection::MM2S)
                          << 8) |
                         channel;
          targets.push_back({t->dma(), key});
        }
      }
      uint64_t deadline = array.cycle() + waitCycles;
      auto satisfied = [&]() {
        for (const Target &t : targets)
          if (t.dma->completedBds(dir, channel) <= tctConsumed[t.key])
            return false;
        return true;
      };
      while (!satisfied() && array.cycle() < deadline)
        array.advance(1);
      if (satisfied()) {
        // One token per target, since the wait is satisfied by one completion
        // from each. Nothing is consumed on a timeout: no token was observed,
        // and spending one would hide the next wait's own stall.
        for (const Target &t : targets)
          ++tctConsumed[t.key];
      } else {
        ++stats.syncTimedOut;
        SyncTimeout rec;
        rec.col = col;
        rec.row = row;
        rec.dir = dir;
        rec.channel = channel;
        rec.targets = static_cast<uint32_t>(targets.size());
        for (const Target &t : targets) {
          uint32_t now = t.dma->completedBds(dir, channel);
          if (now <= tctConsumed[t.key])
            ++rec.unsatisfied;
          rec.completedBefore += tctConsumed[t.key];
          rec.completedAfter += now;
        }
        stats.syncTimeouts.push_back(rec);
      }
      ++stats.sync;
      break;
    }
    case kOpPreempt:
      ++stats.preempt;
      break;
    case kOpLoadPdi:
      // Both switch which CONFIGURATION is loaded rather than doing anything to
      // the array: a preempt saves and restores a context, a load-pdi brings up
      // another one. Neither is followed here, because a caller replays exactly
      // one configuration on purpose -- one CDO group is one hardware context
      // -- and acting on these would replace the design under measurement with
      // a different one, whose CDO was never applied. They are COUNTED rather
      // than ignored, so a sequence that depends on them shows up as a number
      // next to its result instead of as a quietly wrong one.
      ++stats.loadPdi;
      break;
    default:
      break; // operationSize() already rejected anything unrecognised.
    }

    pos += size;
    ++applied;
  }

  if (applied != declaredOps) {
    char buf[112];
    std::snprintf(buf, sizeof(buf),
                  "TXN header declares %u operations but %u were parsed",
                  declaredOps, applied);
    error = buf;
    return TxnOutcome::Malformed;
  }
  return TxnOutcome::Applied;
}

TxnOutcome aiesim::replayTxnFile(Array &array, const std::string &path,
                                 uint64_t base, TxnReplayStats &stats,
                                 std::string &error, uint64_t waitCycles) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    error = "cannot open TXN file: " + path;
    return TxnOutcome::Malformed;
  }
  std::vector<uint8_t> image;
  uint8_t buf[65536];
  while (size_t n = std::fread(buf, 1, sizeof(buf), f))
    image.insert(image.end(), buf, buf + n);
  bool readFailed = std::ferror(f) != 0;
  std::fclose(f);
  if (readFailed) {
    error = "error reading TXN file: " + path;
    return TxnOutcome::Malformed;
  }
  TxnOutcome outcome =
      replayTxn(array, image.data(), image.size(), base, stats, error,
                waitCycles);
  if (outcome != TxnOutcome::Applied)
    error += " [" + path + "]";
  return outcome;
}
