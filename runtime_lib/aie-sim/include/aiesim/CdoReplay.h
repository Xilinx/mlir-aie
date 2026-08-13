//===- CdoReplay.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Replay a design's own CDO configuration blob into the array.
//
// A CDO is the byte-for-byte register traffic the platform loader issues to
// bring a design up: lock initial values, stream-switch routes, buffer
// descriptors, DMA task-queue pushes, the core ELF image, the core enable. So
// replaying one configures the model from what the compiler actually emitted,
// rather than from a hand-written approximation of it -- which is the same
// reason the rest of this simulator is driven through registers instead of a
// side API (Array.h).
//
// This is what a probe cannot substitute for. A hand-programmed BD tests the BD
// the author thought to write; a design's CDO exercises every register its
// toolchain emits, in the order it emits them, including the ones nobody
// remembered to model.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_CDO_REPLAY_H
#define AIESIM_CDO_REPLAY_H

#include "aiesim/Array.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace aiesim {

/// What a replay did, per command kind. Reported rather than asserted: a CDO
/// that turns out to contain no BD writes at all is a scoping fact about the
/// design, and a caller that only counted "success" could not tell that apart
/// from a design whose every BD landed.
struct CdoReplayStats {
  uint32_t write32 = 0;
  uint32_t maskWrite32 = 0;
  uint32_t blockWrite32 = 0; ///< Commands, not words.
  uint64_t blockWriteWords = 0;
  uint32_t maskPoll = 0;
  uint32_t noOp = 0;
  /// Mask-polls whose condition never held within the cycle budget. Not an
  /// error here: a poll waits on array state, and a partially-driven array may
  /// legitimately never reach it. The count is what says so out loud.
  uint32_t maskPollTimedOut = 0;
};

/// Cycles a single CMD_MASK_POLL64 is allowed to spin before it gives up and is
/// counted in `maskPollTimedOut`. The CDO's own timeout field is in
/// milliseconds of wall clock, which has no meaning in a cycle-accurate model,
/// so it is deliberately ignored in favour of a budget the caller controls.
constexpr uint64_t kDefaultMaskPollCycles = 100000;

/// Apply the commands in a CDO image to `array`.
///
/// \param base Added to every command's address. A CDO is emitted against the
///        compiler's own base (AIERT.cpp's XAIE_BASE_ADDR), which need not be
///        the host base this array decodes against; passing the difference is
///        how a design built for one lands on the other.
/// \returns false and sets `error` on a malformed image or an unknown command.
///        A command this function does not recognise stops the replay: the
///        commands after it would be read at the wrong offset, so continuing
///        would apply garbage rather than skip a feature.
bool replayCdo(Array &array, const void *image, size_t sizeBytes, uint64_t base,
               CdoReplayStats &stats, std::string &error,
               uint64_t maskPollCycles = kDefaultMaskPollCycles);

/// As above, reading the image from `path`.
bool replayCdoFile(Array &array, const std::string &path, uint64_t base,
                   CdoReplayStats &stats, std::string &error,
                   uint64_t maskPollCycles = kDefaultMaskPollCycles);

} // namespace aiesim

#endif // AIESIM_CDO_REPLAY_H
