// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Umbrella header. aie-rt provided xaiengine.h; its successor aie-codegen
// provides aie_codegen.h with a different include layout, so mlir-aie keeps the
// xaiengine/ layout and this header. The lite driver is not part of
// aie-codegen.
#ifndef MLIR_AIE_XAIENGINE_H
#define MLIR_AIE_XAIENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <xaiengine/xaie_clock.h>
#include <xaiengine/xaie_core.h>
#include <xaiengine/xaie_dma.h>
#include <xaiengine/xaie_elfloader.h>
#include <xaiengine/xaie_events.h>
#include <xaiengine/xaie_instbuf.h>
#include <xaiengine/xaie_interrupt.h>
#include <xaiengine/xaie_locks.h>
#include <xaiengine/xaie_mem.h>
#include <xaiengine/xaie_noc.h>
#include <xaiengine/xaie_perfcnt.h>
#include <xaiengine/xaie_plif.h>
#include <xaiengine/xaie_reset.h>
#include <xaiengine/xaie_ss.h>
#include <xaiengine/xaie_timer.h>
#include <xaiengine/xaie_trace.h>
#include <xaiengine/xaie_txn.h>
#include <xaiengine/xaiegbl.h>
#include <xaiengine/xaiegbl_defs.h>

#ifdef __cplusplus
}
#endif

#endif
