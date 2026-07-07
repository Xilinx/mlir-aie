//===- AIERuntimeBdLiveness.h -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// kill point is resolved explicitly (the reachable await/free, else region
// end), not read off `mlir::Liveness`.
//
// Why not `mlir::Liveness`? Two reasons. (1) BD-ID hold-range is not SSA
// liveness (see above): the kill is the await/free, not the last use of the
// configure result. (2) `scf.for`/`scf.if` are still structured region ops at
// this stage (no scf->cf lowering has run in the runtime sequence).
// `mlir::Liveness` is region-tolerant but not region-aware: it folds
// nested-region uses into the enclosing block, which would make mutually
// exclusive scf.if arms falsely interfere. This analysis instead uses
// disciplined structural recursion over the region tree.
//
// This runs AFTER --aie-unroll-runtime-sequence-loops, so every constant-trip
// loop is already straight-line. A handle still crossing a loop back-edge
// therefore belongs to a runtime-bound loop (a rolled ping-pong the static
// path cannot lower). `resolveTaskLiveRange` records how many back-edges a live
// handle crosses so the allocator can reject that form for the dynamic path;
// `computePeakBdLiveness` sweeps the sequence for the per-tile peak
// simultaneous liveness the allocator must fit in the tile's BD pool, treating
// scf.if arms as mutually exclusive.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_RUNTIME_BD_LIVENESS_H
#define AIE_RUNTIME_BD_LIVENESS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
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

  /// The completion-sync ops on the terminal handle (`dma_free_task` /
  /// `dma_await_task`), in use order. Empty when no sync is reachable and the
  /// range extends to the end of the sequence (see `leaked`). The first entry is
  /// the kill point; a second entry is the await-then-free idiom (an await that
  /// certifies completion followed by an explicit free of the same handle).
  llvm::SmallVector<mlir::Operation *, 2> syncs;

  /// Number of `scf.for` back-edges the live handle crosses before being freed
  /// (0 = freed in the same iteration as configured, or straight-line). For a
  /// loop that frees the previous iteration's task (ping-pong) this is 1; for
  /// free-two-iterations-ago it is 2; etc. Only meaningful when `!leaked`.
  unsigned backEdgesCrossed = 0;

  /// True when no completion-sync is reachable from the configure: the BD is
  /// held to the end of the sequence. Harmless for straight-line tasks (the ID
  /// simply stays occupied, as for any never-awaited task). But a leaked task
  /// *inside a loop* (see `enclosingLoop`) accumulates one held BD per
  /// iteration and is never reusable across iterations, so the allocator
  /// rejects it.
  bool leaked = false;

  /// Innermost `scf.for` enclosing the configure within the runtime sequence,
  /// or null if the configure is at sequence top level. Used to classify leaks
  /// and loop-carried windows.
  mlir::Operation *enclosingLoop = nullptr;

  /// True when the handle cannot be traced to a single completion sync: it has
  /// more than one carry (e.g. yielded two ways), a carry coexisting with a
  /// sync, a def-use cycle (carried unchanged across a loop back-edge), or a
  /// use the analysis does not understand. Such a task cannot be statically
  /// allocated and must be rejected.
  bool ambiguous = false;

  /// True when tracing the handle followed an `scf.if` yield to the if-result
  /// (the handle escapes its arm via a value join). This case is supported by
  /// the allocator by tracing scf.if yields/results when recycling IDs.
  bool crossedIfJoin = false;
};

/// Number of `aie.dma_bd` ops in a configure task's BD chain (its chain length
/// C, and the count of BD ids it holds). Never zero: a chain with no dma_bd
/// still occupies one id.
unsigned chainLength(DMAConfigureTaskOp configure);

/// Resolve the hold-range of a single configure op by forward-tracing its
/// handle (including across scf.for iter_arg hops) to its completion-sync.
TaskLiveRange resolveTaskLiveRange(DMAConfigureTaskOp configure);

/// Map each completion-sync op (`aiex.dma_await_task` / `aiex.dma_free_task`) in
/// the sequence to the configure op(s) it completes. Built from the same forward
/// handle-trace as `resolveTaskLiveRange`, so the allocator and this analysis
/// share one model of how a task handle flows through `scf.for` / `scf.if`.
///
/// Most syncs map to exactly one configure. Two forms map to several ops:
///   - an `scf.if` value-join free maps to each arm's configure (the arms are
///     mutually exclusive, so only the taken arm actually holds the ids);
///   - the await-then-free idiom maps one configure to both its await and its
///     free op.
/// Ambiguous/leaked configures (which the allocator rejects up front) contribute
/// nothing. A sync op absent from the map completes no configure -- the caller
/// treats that as an unresolved-task error.
llvm::DenseMap<mlir::Operation *, llvm::SmallVector<DMAConfigureTaskOp>>
mapSyncsToConfigures(AIE::RuntimeSequenceOp seq);

/// Compute peak simultaneous BD liveness per tile across a runtime sequence.
/// Keyed by (col, row); the value is the maximum number of BD IDs held at once
/// on that tile, i.e. the window size the allocator must fit in the tile pool.
/// scf.if arms are treated as mutually exclusive; scf.for bodies are swept with
/// loop-carried tasks live (so ping-pong coexistence is counted).
llvm::MapVector<std::pair<int, int>, unsigned>
computePeakBdLiveness(AIE::RuntimeSequenceOp seq);

} // namespace xilinx::AIEX

#endif // AIE_RUNTIME_BD_LIVENESS_H
