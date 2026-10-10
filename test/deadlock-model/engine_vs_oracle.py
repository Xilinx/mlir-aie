#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s --seeds 150 | FileCheck %s

"""--aie-check-deadlock against the deadlock model's oracle, on small
generated designs inside the subset the pass decides (aiemodel.subset).

Wherever the pass decides, the oracle, which tries every order, has to agree:
- accepted: the oracle accepts with no buffering;
- a deadlock: the oracle deadlocks even with the pass's buffering in front of
  each receiver;
- a deadlock only without buffering: the oracle deadlocks without and
  accepts with it.
Designs the pass leaves undecided, or the oracle finds too large, are counted.
"""

import argparse
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
# The model lives with the router's property tests.
sys.path.insert(0, str(HERE.parent / "create-packet-flows" / "nightly"))
from aiemodel.program import explore, load_system  # noqa: E402
from aiemodel.run import aie_opt_path  # noqa: E402
from aiemodel.subset import subset_design  # noqa: E402

BUFFERING = 8


def engine(text):
    """The pass's verdict per runtime sequence name."""
    p = subprocess.run(
        [aie_opt_path(), "--aie-check-deadlock=allow-undecided", "-o", "/dev/null"],
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    verdicts = {}
    for line in p.stderr.splitlines():
        m = re.search(r"(error|warning): (.*)", line)
        if not m:
            continue
        msg = m.group(2)
        seq = re.search(r"@(\w+)", msg)
        name = seq.group(1) if seq else None
        if msg.startswith("cannot decide"):
            verdicts[name] = "outside"
        elif "deadlocks unless" in msg:
            verdicts[name] = "buffered"
        elif "deadlocks" in msg:
            verdicts[name] = "deadlock"
        elif "still in flight" in msg:
            verdicts[name] = "unquiesced"
    return p.returncode, verdicts


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--seeds", type=int, default=150)
    cli.add_argument("--first-seed", type=int, default=0)
    cli.add_argument("--max-states", type=int, default=30000)
    args = cli.parse_args()
    counts = Counter()
    wrong = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        text = subset_design(random.Random(seed))
        rc, verdicts = engine(text)
        system = load_system(text)
        for seq in system.sequences or [None]:
            verdict = verdicts.get(seq, "accept")
            counts[verdict] += 1
            if verdict in ("outside", "unquiesced"):
                continue
            least = explore(system, seq, 0, args.max_states)
            if least.outcome == "undecided":
                counts["oracle too large"] += 1
                continue
            most = explore(system, seq, BUFFERING, args.max_states)
            agree = {
                "accept": least.outcome in ("accept", "unquiesced"),
                "deadlock": least.outcome == "deadlock"
                and most.outcome in ("deadlock", "undecided"),
                "buffered": least.outcome == "deadlock"
                and most.outcome in ("accept", "unquiesced", "undecided"),
            }[verdict]
            counts["checked"] += 1
            if not agree:
                wrong.append(
                    f"seed {seed} @{seq}: pass says {verdict}, oracle "
                    f"{least.outcome} without buffering, {most.outcome} with"
                )
    for line in wrong:
        print("WRONG:", line)
    print(
        "engine-vs-oracle: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f", {len(wrong)} disagree"
    )


# CHECK-NOT: WRONG
# CHECK: engine-vs-oracle: {{.*}}, 0 disagree

if __name__ == "__main__":
    main()
