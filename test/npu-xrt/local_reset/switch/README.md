<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Stream-switch connection reset

The stream switch has no reset intrinsic, so a "reset" is a **config-disable**:
clear a port-config register, then write it back to re-establish the route.

Tile `(0, 2)` streams a resident buffer to the host over a **run-forever** MM2S
routed `DMA:0 -> South` toward the shim. The single BD loops back to itself
(`next_bd`), lock-gated so that between sends the channel is **stalled on the lock
acquire** and the routed datapath is quiescent -- switch-config writes take effect
on a settled datapath, not mid-stream.

Each dispatch re-enables the `DMA:0` slave port
(`Stream_Switch_Slave_DMA_0_Config`, `Slave_Enable` at bit 31; enabled value
`0x80000000`), re-arms the lock, collects, then disables the port again at the
end. The config register is sticky, so the port stays off between dispatches and
the re-enable at the start of the next dispatch genuinely restores the route --
rather than being a no-op net of an adjacent disable. The buffer is resident, so
every dispatch must return the same bytes (`100 + i`).

Only the slave port is toggled -- the master (South) config is left resident -- so
a failure pins to that one port rather than to the DMA channel or the route as a
whole. The DMA channel itself is never reset here (that is the `dma` family).

The `DMA:0` slave-config offset (`0x3F104`) and the lock re-arm `LOCK0_VALUE`
(`0x1F000`) are the same on npu1 (AIE-ML) and npu2 (AIE2P), so this runs on both.
See [`../README.md`](../README.md) for the shared design, the architecture note,
and how to run.

## Behaviour

The disable and re-enable straddle the dispatch boundary -- re-enable at the
start, disable at the end -- so the port is genuinely off at the start of every
dispatch after the first, keeping the re-enable on the critical path:

- **Correct protocol (this test):** re-enable the port, re-arm the lock, collect,
  then disable -> every dispatch returns the resident buffer.
- **Omit the re-enable:** dispatch 0 still streams (port enabled from CDO load),
  but its end-disable leaves the port off, so dispatch 1 has nowhere to stream and
  does not complete.
- **Omit both switch writes:** the route stays resident and the lock re-arm alone
  gates each send, so it passes -- which is exactly why the disable is included, so
  that the re-enable actually exercises the port instead of being a net no-op.

Reproduce the failure mode by hand by deleting the re-enable `write32` in
`aie.mlir`.

## Reference

The slave-port config register, its `Slave_Enable` bit, and the connection
enable/disable procedure are defined in the public aie-rt driver
(<https://github.com/Xilinx/aie-rt>, vendored at `third_party/aie-rt/`):
`XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0` (`_SLAVE_ENABLE_LSB`=31)
and `..._LOCK0_VALUE` in `driver/src/global/xaie2pgbl_params.h`;
`XAie_StrmConnCctEnable` / `XAie_StrmConnCctDisable` in
`driver/src/stream_switch/xaie_ss.c`; `XAie_LockSetValue` in
`driver/src/locks/xaie_locks.c`. The npu1 (AIE-ML) equivalent is
`XAIEMLGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0` at the same `0x3F104` in
`xaiemlgbl_params.h`. See [`../README.md`](../README.md#references) for the full
table.
