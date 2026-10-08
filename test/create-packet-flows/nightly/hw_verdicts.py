#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

"""The deadlock rules router_properties.py checks the router with, against
what happened on hardware.

Each file in Inputs/hw_verdicts is a design as the router got it, then, after
`// -----`, the routing that ran on an NPU2 (Strix): this router's, the one
the router at main or an earlier version of this one made (`_main`, `_old`),
or one forced onto a shared arbiter or link. The prio_ designs carry the
routes aiecc's column control overlay adds for the task-complete tokens. Its
first line says whether that run passed or hung. Every routing that hung must
break a rule; every one that passed must not, unless it is marked cautious:
what kept it from hanging is timing, buffering or a program hidden from the
router, none of which the rules can see.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import router_properties as rp  # noqa: E402

UNSAFE = ("conflicting", "hold cycle")


def main():
    cases = sorted((Path(__file__).parent / "Inputs" / "hw_verdicts").glob("*.mlir"))
    wrong = []
    for path in cases:
        text = path.read_text()
        head = {}
        for line in text.splitlines():
            key, _, value = line.removeprefix("// ").partition(": ")
            if key not in ("HW", "hops", "cautious"):
                break
            head[key] = value
        src, routed = text.split("\n// -----\n")
        d = rp.load_design(src)
        hops_on = head.get("hops") != "off"
        problems, stats = rp.verify(d, rp.Analysis(d), routed, hops_on)
        unsafe = [p for p in problems if p.startswith(UNSAFE)]
        if stats["shared_receiver_cycle"]:
            unsafe.append(
                "hold cycle (shared receiver): " + stats["shared_receiver_cycle"]
            )
        other = [p for p in problems if not p.startswith(UNSAFE)]
        want = head["HW"] == "HANG" or "cautious" in head
        if bool(unsafe) != want or other:
            wrong.append(f"{path.stem}: HW {head['HW']}, {problems or 'no problems'}")
    for line in wrong:
        print("WRONG:", line)
    print(f"hw-verdicts: {len(cases)} routings, {len(wrong)} disagree with HW")


# CHECK-NOT: WRONG
# CHECK: hw-verdicts: 133 routings, 0 disagree with HW

if __name__ == "__main__":
    main()
