#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s
# RUN: sed 's/NPUDEVICE/npu2_1col/' %S/../npu-xrt/objectfifo_pack_shim_sharing/aie.mlir | aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform > %t.mlir
# RUN: %python %s --design %t.mlir | FileCheck %s --check-prefix=LOWERED

"""The deadlock model's oracle (aiemodel.program) on designs worked by hand,
and on an objectFifo design lowered by aie-opt.

Each case names a design in Inputs, the runtime sequence to dispatch, the
words of buffering in front of each receiver, and what docs/DeadlockModel.md
says of it.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
# The model lives with the router's property tests.
sys.path.insert(0, str(HERE.parent / "create-packet-flows" / "nightly"))
from aiemodel.program import explore, load_system  # noqa: E402

CASES = [
    # A head-of-line wait: to_a, issued first, cannot finish without to_b,
    # queued behind it on the same channel.
    ("head_of_line", "first_a", 0, "deadlock"),
    # Enough buffering in front of the receiver to hold the rest of to_a
    # lets it finish, which is why the model decides with the least
    # buffering a path has.
    ("head_of_line", "first_a", 4, "accept"),
    # Issued the other way round, to_b goes first and nothing waits.
    ("head_of_line", "first_b", 0, "accept"),
    # Two acquirers race for one lock: outside the subset the model decides.
    ("contended_lock", "run", 0, "undecided"),
    # A channel the design does not program could do anything.
    ("hidden_channel", "run", 0, "undecided"),
    # The dispatch ends with a transfer still in flight.
    ("unquiesced", "run", 0, "unquiesced"),
]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--design", type=Path, help="decide each sequence of this design")
    args = cli.parse_args()
    if args.design:
        # A lowered objectFifo design: loop-carried lock amounts, packet
        # headers set by aie.dma_bd_packet, and a shim channel two members
        # of a pack take turns on.
        system = load_system(args.design.read_text())
        for sequence in system.sequences:
            v = explore(system, sequence)
            print(f"{sequence}: {v.outcome} {v.reason}")
        return
    for name, sequence, capacity, want in CASES:
        system = load_system((HERE / "Inputs" / f"{name}.mlir").read_text())
        v = explore(system, sequence, capacity)
        detail = v.reason or ", ".join(v.unquiesced) or f"{v.states} states"
        ok = "OK" if v.outcome == want else f"WRONG, want {want}"
        print(f"{name} {sequence} capacity {capacity}: {v.outcome} ({detail}) {ok}")
        if v.outcome == "deadlock":
            print("  last moves:", "; ".join(v.schedule[-3:]))


# CHECK: head_of_line first_a capacity 0: deadlock ({{.*}}) OK
# CHECK-NEXT: last moves: {{.*}}acquires (2, 'prod1', 1)
# CHECK: head_of_line first_a capacity 4: accept ({{.*}}) OK
# CHECK: head_of_line first_b capacity 0: accept ({{.*}}) OK
# CHECK: contended_lock run capacity 0: undecided (some orders deadlock and some finish{{.*}}) OK
# CHECK: hidden_channel run capacity 0: undecided (a flow names ((0, 1), 0, 0), whose channel the design does not program) OK
# CHECK: unquiesced run capacity 0: unquiesced (((0, 0), 1, 0) still has work) OK
# CHECK-NOT: WRONG

# LOWERED: run_a: accept
# LOWERED: run_b: accept
# LOWERED: run_c: accept

if __name__ == "__main__":
    main()
