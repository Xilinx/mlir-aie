//===- AIERuntimeBdLiveness.h -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Control-flow-aware liveness analysis for runtime-sequence DMA tasks, used to
// drive buffer-descriptor (BD) ID allocation as graph coloring.
//
// A BD ID models hardware state: it is held from the point a task is configured
// (`aiex.dma_configure_task`, which defines an Index SSA value) until the DMA
// engine has finished executing it, signalled by `aiex.dma_await_task` /
// `aiex.dma_free_task`. This hold-range is NOT the same as SSA-value liveness
// of the configure result: a configure followed only by `dma_start_task` has
// its last *use* at the start, yet the BD is still physically in flight. So the
// kill point is resolved explicitly (free / synthesized-after-await /
// region-end), not read off `mlir::Liveness`.
//
// Because `scf.for`/`scf.if` are still structured region ops at this stage (no
// scf->cf lowering has run), this analysis uses disciplined structural
// recursion over the region tree rather than CFG-based `mlir::Liveness` (which
// is region- tolerant but not region-aware: it folds nested-region uses into
// the enclosing block and would falsely make scf.if arms interfere).
//
// Loop-carried tasks: a handle freed in a *later* loop iteration than it was
// configured (the ping-pong "free the previous iteration" pattern, expressible
// via `scf.for` iter_args) needs more than one physical BD ID — a rotating
// window. The analysis records how many loop back-edges the live handle crosses;
// the window size (peak simultaneous liveness) is computed by the BD-ID
// allocator from the interfering set of ranges, not here.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_RUNTIME_BD_LIVENESS_H
#define AIE_RUNTIME_BD_LIVENESS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {

/// The resolved hold-range of a single `aiex.dma_configure_task`.
///
/// A BD ID is held from `configure` until the hardware certifies the transfer
/// complete, i.e. the matching `dma_await_task`/`dma_free_task`. This is found
/// by forward-tracing the configure's Index result, following `scf.for`
/// iter_arg carries across loop back-edges. The result describes *what* must be
/// allocated, independent of *how* the BD id(s) are written back onto the IR
/// (which depends on the not-yet-settled dynamic bd_id operand form).
struct TaskLiveRange {
  /// The configure op whose BD chain needs IDs.
  DMAConfigureTaskOp configure;

  /// The op that ends this task's hold-range (an explicit `dma_free_task` or a
  /// `dma_await_task`), or null when no completion-sync is reachable and the
  /// range extends to the end of the runtime sequence (see `leaked`).
  mlir::Operation *kill = nullptr;

  /// Number of `scf.for` back-edges the live handle crosses before being freed
  /// (0 = freed in the same iteration as configured, or straight-line). For a
  /// loop that frees the previous iteration's task (ping-pong) this is 1; for
  /// free-two-iterations-ago it is 2; etc. Only meaningful when `!leaked`.
  unsigned backEdgesCrossed = 0;

  /// True when no completion-sync is reachable from the configure: the BD is
  /// held to the end of the sequence. Harmless for straight-line tasks (the ID
  /// simply stays occupied, as for any never-awaited task). But a leaked task
  /// *inside a loop* (see `enclosingLoop`) accumulates one held BD per
  /// iteration -- bounded (allocatable as a window) only if the loop trip count
  /// is a compile-time constant; otherwise unbounded and unallocatable.
  bool leaked = false;

  /// Innermost `scf.for` enclosing the configure within the runtime sequence,
  /// or null if the configure is at sequence top level. Used to classify leaks
  /// and loop-carried windows.
  mlir::Operation *enclosingLoop = nullptr;
};

/// Resolve the hold-range of a single configure op by forward-tracing its
/// handle (including across scf.for iter_arg hops) to its completion-sync.
TaskLiveRange resolveTaskLiveRange(DMAConfigureTaskOp configure);

} // namespace xilinx::AIEX

#endif // AIE_RUNTIME_BD_LIVENESS_H
