//===- TxnReplay.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Replay a design's runtime instruction sequence -- the TXN binary.
//
// This is the OTHER half of a design's configuration, and the half that makes
// it run. [[CdoReplay.h]] brings the array up: locks, routes, core-tile and
// memtile BDs, the core images, the enables. What it does NOT contain is any
// shim BD, because the shim is where host buffers enter and their addresses are
// not known until the host binds them. Those BDs, and the task pushes that
// start them, live here -- so a design replayed from its CDO alone has every
// core waiting on data that nothing was ever going to fetch.
//
// A decoded example, the whole of op0_LayerNorm_sequence.bin, which is what
// this file exists to apply:
//
//   BLOCKWRITE 0x1D000  8 words   shim BD 0 (length, flags; address left 0)
//   DDR_PATCH  0x1D004  arg 0     ... and its address, from kernel argument 0
//   WRITE      0x1D214  0         push BD 0 on MM2S channel 0
//   BLOCKWRITE 0x1D020  8 words   shim BD 1
//   DDR_PATCH  0x1D024  arg 1     ... from kernel argument 1
//   MASKWRITE  0x1D200            S2MM channel 0 control
//   WRITE      0x1D204  0x80000001 push BD 1 on S2MM channel 0, token enabled
//   TCT        col 0 ch 0 S2MM    wait for it to finish
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_TXN_REPLAY_H
#define AIESIM_TXN_REPLAY_H

#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aiesim {

/// Where the model puts the host buffer behind kernel argument N.
///
/// A TXN sequence never contains a buffer address; it contains a PATCH saying
/// "this BD's address word comes from kernel argument N, plus this offset". On
/// hardware the runtime fills those in from the buffer objects the host bound.
/// Nothing bound any here, so the model has to choose -- and what it chooses is
/// visible rather than hidden, because every address the design then reads or
/// writes is derived from it.
///
/// One argument per 4 GiB, so argument N is simply the address range with N in
/// its high half. That is deliberately far more room than any buffer needs, and
/// it buys two things worth more than compactness: no two arguments can ever
/// alias no matter how large an offset a design applies, and the offset lands
/// unchanged in the low word, so a patched BD address reads back as its own
/// argument index and offset. The model's DDR is sparse and page-backed
/// (Array::ddrRead), so the empty space between them costs nothing.
///
/// This makes buffer IDENTITY exact and buffer CONTENT arbitrary: a design's
/// input buffers are whatever the model's zero-filled DDR holds unless a caller
/// writes them. Byte movement and completion are therefore trustworthy;
/// computed VALUES are not, and no caller should read them as if they were.
constexpr uint64_t kArgumentStride = 1ULL << 32;
inline uint64_t argumentBase(uint32_t argIdx) {
  return static_cast<uint64_t>(argIdx) * kArgumentStride;
}

/// A TCT wait that gave up, and what it was waiting on.
///
/// Kept per occurrence rather than only counted: a design's waits are not
/// interchangeable, and "4 of 8 timed out" names neither which channel stalled
/// nor whether the whole wait or one tile of it was the problem. Same reason the
/// DMA diagnostics name their tile and BD -- a fault somewhere is not a location.
struct SyncTimeout {
  /// The tile the descriptor names, which is where the wait's rectangle starts.
  uint32_t col = 0, row = 0;
  DmaDirection dir = DmaDirection::S2MM;
  uint32_t channel = 0;
  /// Tiles in the wait's rectangle that exist and have a DMA, and how many of
  /// those never completed a BD on that channel. `unsatisfied < targets` is a
  /// PARTIAL stall -- some tiles delivered and others did not, which is a
  /// different fault from a channel nothing ever ran.
  uint32_t targets = 0, unsatisfied = 0;
  /// Completions already on the counter when the wait STARTED, summed over its
  /// targets, and the same sum when it gave up. Equal and non-zero means the
  /// BD this wait exists for had already completed before the wait began -- so
  /// the wait is asking for a SECOND completion that was never coming, which is
  /// a different fault from a channel that never ran.
  uint32_t completedBefore = 0, completedAfter = 0;
};

/// What a replay did, per opcode. Reported rather than asserted, for the reason
/// CdoReplayStats gives: a sequence with no shim BD in it is a fact about the
/// design, not a failure, and a caller that only saw "ok" could not tell.
struct TxnReplayStats {
  uint32_t write32 = 0;
  uint32_t maskWrite32 = 0;
  uint32_t blockWrite32 = 0; ///< Commands, not words.
  uint64_t blockWriteWords = 0;
  uint32_t maskPoll = 0;
  uint32_t addressPatch = 0;
  /// Patches whose target word did NOT already hold the offset the patch
  /// carries. Zero everywhere the model's register state is trustworthy, which
  /// is what makes "replace" and "add" the same computation there; see
  /// TxnReplay.cpp. Non-zero says the two readings have diverged, and any
  /// address in that run depends on which one is right.
  uint32_t addressPatchPreloadMismatch = 0;
  uint32_t sync = 0; ///< Task-completion-token waits (TCT).
  uint32_t loadPdi = 0;
  uint32_t preempt = 0;
  /// Waits whose condition never held within the cycle budget. Not an error: a
  /// wait depends on array state, and a model with an unmodelled corner may
  /// legitimately never reach it. The count is what says so out loud.
  uint32_t maskPollTimedOut = 0;
  uint32_t syncTimedOut = 0;
  /// One entry per timed-out TCT, in sequence order. Size equals syncTimedOut.
  std::vector<SyncTimeout> syncTimeouts;
};

/// Cycles one wait (a MASKPOLL or a TCT) may spin before it gives up and is
/// counted as timed out. A TXN carries no cycle budget of its own -- on
/// hardware these block until the device answers -- so the budget is the
/// caller's to set.
constexpr uint64_t kDefaultTxnWaitCycles = 100000;

/// What a replay attempt did with the image as a whole.
enum class TxnOutcome {
  Applied,
  /// The image is well formed but describes a run this model is not
  /// performing, so nothing was applied. Distinct from Malformed because the
  /// caller's design is still measurable -- it just was not driven by this
  /// sequence, and a report that conflated the two would read as a parse bug.
  Declined,
  Malformed,
};

/// Apply the operations in a TXN image to `array`.
///
/// \param base Added to every register address, exactly as replayCdo takes it
///        and for the same reason: a sequence is emitted against the
///        compiler-side device instance, whose base is zero, while this array
///        decodes against the host base its DeviceModel carries. Without it
///        every address falls below the base and decodes as unmapped.
/// \returns Malformed, with `error` set, on a bad header or an unknown opcode.
///        An opcode this function does not recognise STOPS the replay rather
///        than being skipped: every operation carries its own size, so an
///        unknown one leaves no way to find the next, and continuing would
///        apply garbage rather than omit a feature. Same rule as replayCdo.
///
///        Declined, with `error` set to why, for a sequence that loads its own
///        configurations -- see the load-pdi handling in TxnReplay.cpp. Nothing
///        is applied in that case, so the array is untouched rather than half
///        programmed.
TxnOutcome replayTxn(Array &array, const void *image, size_t sizeBytes,
                     uint64_t base, TxnReplayStats &stats, std::string &error,
                     uint64_t waitCycles = kDefaultTxnWaitCycles);

/// As above, reading the image from `path`.
TxnOutcome replayTxnFile(Array &array, const std::string &path, uint64_t base,
                         TxnReplayStats &stats, std::string &error,
                         uint64_t waitCycles = kDefaultTxnWaitCycles);

} // namespace aiesim

#endif // AIESIM_TXN_REPLAY_H
