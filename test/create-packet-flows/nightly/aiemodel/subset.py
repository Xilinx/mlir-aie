#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Small designs inside the deadlock model's deterministic subset, for
checking the deadlock engine against the oracle.

Cores on one column take inputs and give outputs through lock-paced buffers
on their DMA channels. Each input comes from the shim (sometimes two inputs
share one shim channel by packet id, the head-of-line case) or from another
core's output; each output goes to the shim or to another core. Cores
acquire their inputs in varied orders, and the runtime sequence issues its
transfers and waits in varied orders, with varied lengths, so some designs
finish, some deadlock, and some leave work in flight.
"""

import random


def subset_design(rng: random.Random) -> str:
    cores = rng.randint(1, 3)
    tiles = [(0, 2 + i) for i in range(cores)]
    trips = rng.randint(1, 3)
    lines = ["module {", "  aie.device(npu2_1col) {", "    %s = aie.tile(0, 0)"]
    lines += [f"    %t{i} = aie.tile(0, {r})" for i, (_, r) in enumerate(tiles)]
    # Per core: its input buffers (S2MM channels) and its output (MM2S 0).
    ins = {i: rng.randint(1, 2) for i in range(cores)}
    has_out = {i: rng.random() < 0.8 for i in range(cores)}
    words = {}
    depth = {}
    body = []
    mems = []
    for i in range(cores):
        ports = [("in", j) for j in range(ins[i])]
        if has_out[i]:
            ports.append(("out", 0))
        mem = [
            f"    %mem{i} = aie.mem(%t{i}) {{",
            "      %one = arith.constant 1 : i32",
        ]
        for k, (kind, j) in enumerate(ports):
            name = f"{kind}{i}_{j}"
            w = rng.randint(1, 4)
            d = rng.randint(1, 2)
            words[name], depth[name] = w, d
            for b in range(d):
                body.append(f"    %{name}_b{b} = aie.buffer(%t{i}) : memref<{w}xi32>")
            body.append(
                f'    %{name}_p = aie.lock(%t{i}) {{init = {d} : i32, sym_name = "{name}_p"}}'
            )
            body.append(
                f'    %{name}_c = aie.lock(%t{i}) {{init = 0 : i32, sym_name = "{name}_c"}}'
            )
            dir_ = "S2MM" if kind == "in" else "MM2S"
            nxt = f"^s{k + 1}" if k + 1 < len(ports) else "^end"
            # S2MM fills (acquire p, release c); MM2S drains (acquire c, release p).
            acq, rel = (
                (f"{name}_p", f"{name}_c")
                if kind == "in"
                else (f"{name}_c", f"{name}_p")
            )
            if k:
                mem.append(f"    ^s{k}:")
            mem.append(f"      %d{k} = aie.dma_start({dir_}, {j}, ^{name}_0, {nxt})")
            for b in range(d):
                after = f"^{name}_{b + 1}" if b + 1 < d else f"^{name}_0"
                mem += [
                    f"    ^{name}_{b}:",
                    f"      aie.use_lock(%{acq}, AcquireGreaterEqual, %one)",
                    f"      aie.dma_bd(%{name}_b{b} : memref<{w}xi32> len = {w})",
                    f"      aie.use_lock(%{rel}, Release, %one)",
                    f"      aie.next_bd {after}",
                ]
        mem += ["    ^end:", "      aie.end", "    }"]
        mems += mem
    # Wiring: each input from the shim or from another core's output.
    free_outs = [i for i in range(cores) if has_out[i]]
    rng.shuffle(free_outs)
    flows, shim_ins, shim_outs = [], [], []
    for i in range(cores):
        for j in range(ins[i]):
            src = next((o for o in free_outs if o != i), None)
            if src is not None and rng.random() < 0.4:
                free_outs.remove(src)
                flows.append(f"    aie.flow(%t{src}, DMA : 0, %t{i}, DMA : {j})")
            else:
                shim_ins.append((i, j))
    for o in free_outs:
        shim_outs.append(o)
    # Shim inputs: on two MM2S channels, sometimes shared by packet id.
    allocs, transfers = [], []
    for k, (i, j) in enumerate(shim_ins):
        allocs.append((f"in{i}_{j}", k % 2 if rng.random() < 0.5 else 0))
    by_channel = {}
    for name, ch in allocs:
        by_channel.setdefault(ch, []).append(name)
    pid = 0
    for ch, names in sorted(by_channel.items()):
        packet = len(names) > 1
        for name in names:
            i, j = (int(x) for x in name[2:].split("_"))
            if packet:
                lines.append(
                    f"    aie.shim_dma_allocation @{name}_a(%s, MM2S, {ch}, <pkt_id = {pid}>)"
                )
                flows.append(
                    f"    aie.packet_flow({pid}) {{ aie.packet_source<%s, DMA : {ch}> "
                    f"aie.packet_dest<%t{i}, DMA : {j}> }}"
                )
                pid += 1
            else:
                lines.append(f"    aie.shim_dma_allocation @{name}_a(%s, MM2S, {ch})")
                flows.append(f"    aie.flow(%s, DMA : {ch}, %t{i}, DMA : {j})")
            # How many words to send: what the core takes, sometimes not.
            want = words[name] * trips
            n = (
                want
                if rng.random() < 0.8
                else max(1, want + rng.choice([-1, 1]) * words[name])
            )
            transfers.append(("in", name, n))
    for k, o in enumerate(shim_outs[:2]):
        name = f"out{o}_0"
        lines.append(f"    aie.shim_dma_allocation @{name}_a(%s, S2MM, {k})")
        flows.append(f"    aie.flow(%t{o}, DMA : 0, %s, DMA : {k})")
        transfers.append(("out", name, words[name] * trips))
    for o in shim_outs[2:]:
        has_out[o] = False  # no shim channel left; the core keeps its output
    lines += body + flows + mems
    # Cores: per trip, acquire inputs in a random order, then the output.
    for i in range(cores):
        steps = []
        order = list(range(ins[i]))
        for _ in range(trips):
            rng.shuffle(order)
            for j in order:
                steps.append(
                    f"      aie.use_lock(%in{i}_{j}_c, AcquireGreaterEqual, %one)"
                )
            if has_out[i]:
                steps.append(
                    f"      aie.use_lock(%out{i}_0_p, AcquireGreaterEqual, %one)"
                )
            for j in order:
                steps.append(f"      aie.use_lock(%in{i}_{j}_p, Release, %one)")
            if has_out[i]:
                steps.append(f"      aie.use_lock(%out{i}_0_c, Release, %one)")
        lines += [
            f"    %core{i} = aie.core(%t{i}) {{",
            "      %one = arith.constant 1 : i32",
        ]
        lines += steps + ["      aie.end", "    }"]
    # The runtime sequence: transfers in a random order, waits on outputs.
    rng.shuffle(transfers)
    seq = ["    aie.runtime_sequence @run(%x : memref<64xi32>) {"]
    waits = []
    for n_, (kind, name, n) in enumerate(transfers):
        token = ", issue_token = true" if kind == "out" else ""
        seq.append(
            f"      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, {n}][0, 0, 0, 1]) "
            f"{{metadata = @{name}_a, id = {n_} : i64{token}}} : memref<64xi32>"
        )
        if kind == "out":
            waits.append(name)
        if waits and rng.random() < 0.3:
            seq.append(f"      aiex.npu.dma_wait {{symbol = @{waits.pop(0)}_a}}")
    rng.shuffle(waits)
    seq += [f"      aiex.npu.dma_wait {{symbol = @{w}_a}}" for w in waits]
    seq.append("    }")
    lines += seq + ["  }", "}"]
    return "\n".join(lines) + "\n"
