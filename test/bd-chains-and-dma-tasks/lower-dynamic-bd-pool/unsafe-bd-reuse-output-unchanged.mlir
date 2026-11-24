// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// warn-unsafe-bd-reuse changes diagnostics only. Off, the queue-depth guard
// stops modelling configures, frees and completion, so each design here is
// lowered through both BD passes, as aiecc runs them, with the flag on and off
// and the two outputs must match. Between them the designs exhaust the
// ownership budget, converge loops, cross unknown regions and insert polls.
//
// DEFINE: %{on} = aie-opt --split-input-file --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true'
// DEFINE: %{off} = aie-opt --split-input-file --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=false' --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=false'
//
// RUN: rm -rf %t.on %t.off && mkdir -p %t.on %t.off
// RUN: %{on} %S/unsafe-bd-reuse-state-budget.mlir -o %t.on/budget 2> %t.budget.err
// RUN: %{off} %S/unsafe-bd-reuse-state-budget.mlir -o %t.off/budget 2> %t.budget.quiet
// RUN: FileCheck %s --check-prefix=BUDGET < %t.budget.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.budget.quiet
//
// RUN: %{on} %S/unsafe-bd-reuse-state-convergence.mlir -o %t.on/converge
// RUN: %{off} %S/unsafe-bd-reuse-state-convergence.mlir -o %t.off/converge
// RUN: FileCheck %s --check-prefix=POLL < %t.on/converge
//
// RUN: %{on} %S/unsafe-bd-reuse-unknown-region.mlir -o %t.on/unknown 2> %t.unknown.err
// RUN: %{off} %S/unsafe-bd-reuse-unknown-region.mlir -o %t.off/unknown 2> %t.unknown.quiet
// RUN: FileCheck %s --check-prefix=WARN < %t.unknown.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.unknown.quiet
// RUN: FileCheck %s --check-prefix=POLL < %t.on/unknown
//
// RUN: %{on} %S/unsafe-bd-reuse-ownership.mlir -o %t.on/owner 2> %t.owner.err
// RUN: %{off} %S/unsafe-bd-reuse-ownership.mlir -o %t.off/owner 2> %t.owner.quiet
// RUN: FileCheck %s --check-prefix=WARN < %t.owner.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.owner.quiet
//
// RUN: %{on} %S/unsafe-bd-reuse-polls.mlir -o %t.on/polls 2> %t.polls.err
// RUN: %{off} %S/unsafe-bd-reuse-polls.mlir -o %t.off/polls 2> %t.polls.quiet
// RUN: FileCheck %s --check-prefix=WARN < %t.polls.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.polls.quiet
// RUN: FileCheck %s --check-prefix=POLL < %t.on/polls
//
// RUN: diff -r %t.on %t.off

// BUDGET: warning: frees a task on tile (0,0) MM2S channel 0 with no completion guarantee
// BUDGET: note: task ownership analysis budget exhausted
// WARN: warning: frees a task on tile (0,0) MM2S channel 0 with no completion guarantee
// QUIET-NOT: warning: frees a task
// POLL: aiex.npu.maskpoll
