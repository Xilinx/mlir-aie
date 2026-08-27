//===- push_bd0.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Re-arms this tile's own MM2S channel 0 by pushing BD 0 to its hardware
// task queue -- a plain volatile store to a fixed local (unfolded) address,
// exactly the idiom test/npu-xrt/program_memory_overlay's `_poll_ctrl_done`
// stub already uses for a different fixed hardware address (a lock's
// register, there; a DMA channel's task-queue register, here).
//
// Address is XAIE2PGBL_MEMORY_MODULE_DMA_MM2S_0_START_QUEUE (0x1DE10) plus
// one word (StartBd.Idx = 1 in aie-rt's Aie2PTileDmaProp) = 0x1DE14.
// Verified against aie-rt's own encoding via `aie-translate --aie-generate-cdo
// --cdo-debug=true` on a throwaway single-BD probe: pushing bd_id=N there
// prints a plain `Write64: Address: ...DE14  Data: N` with no other bits set
// -- XAie_DmaChannelPushBdToQueue() writes the raw BD number unshifted, not
// through the Lsb/Mask fields RptCount/EnToken also use.
extern "C" void push_bd0(void) {
  volatile uint32_t *start_queue = (volatile uint32_t *)0x0001DE14;
  *start_queue = 0;
}
