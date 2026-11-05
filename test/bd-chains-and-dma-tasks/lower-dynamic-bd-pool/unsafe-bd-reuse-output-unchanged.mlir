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
// RUN: %{on} %S/unsafe-bd-reuse-state-budget.mlir -o %t.budget.on 2> %t.budget.err
// RUN: %{off} %S/unsafe-bd-reuse-state-budget.mlir -o %t.budget.off 2> %t.budget.quiet
// RUN: diff %t.budget.on %t.budget.off
// RUN: FileCheck %s --check-prefix=BUDGET < %t.budget.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.budget.quiet
//
// RUN: %{on} %S/unsafe-bd-reuse-state-convergence.mlir -o %t.converge.on
// RUN: %{off} %S/unsafe-bd-reuse-state-convergence.mlir -o %t.converge.off
// RUN: diff %t.converge.on %t.converge.off
// RUN: FileCheck %s --check-prefix=POLL < %t.converge.on
//
// RUN: %{on} %S/unsafe-bd-reuse-unknown-region.mlir -o %t.unknown.on 2> %t.unknown.err
// RUN: %{off} %S/unsafe-bd-reuse-unknown-region.mlir -o %t.unknown.off 2> %t.unknown.quiet
// RUN: diff %t.unknown.on %t.unknown.off
// RUN: FileCheck %s --check-prefix=WARN < %t.unknown.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.unknown.quiet
// RUN: FileCheck %s --check-prefix=POLL < %t.unknown.on
//
// RUN: %{on} %S/unsafe-bd-reuse-ownership.mlir -o %t.owner.on 2> %t.owner.err
// RUN: %{off} %S/unsafe-bd-reuse-ownership.mlir -o %t.owner.off 2> %t.owner.quiet
// RUN: diff %t.owner.on %t.owner.off
// RUN: FileCheck %s --check-prefix=WARN < %t.owner.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.owner.quiet
//
// RUN: %{on} %S/unsafe-bd-reuse-polls.mlir -o %t.polls.on 2> %t.polls.err
// RUN: %{off} %S/unsafe-bd-reuse-polls.mlir -o %t.polls.off 2> %t.polls.quiet
// RUN: diff %t.polls.on %t.polls.off
// RUN: FileCheck %s --check-prefix=WARN < %t.polls.err
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty < %t.polls.quiet
// RUN: FileCheck %s --check-prefix=POLL < %t.polls.on

// BUDGET: warning: frees a task on tile (0,0) MM2S channel 0 with no completion guarantee
// BUDGET: note: task ownership analysis budget exhausted
// WARN: warning: frees a task on tile (0,0) MM2S channel 0 with no completion guarantee
// QUIET-NOT: warning: frees a task
// POLL: aiex.npu.maskpoll
