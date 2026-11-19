#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Write the cores a lock test's self-loop pools need, which its text leaves out.

The lock tests make lock demand from self-loop pools,
``aie.objectfifo @f(%t, {%t}, depth)``. aie-objectfifo-split needs a filler and
a drainer on each pool, and the core verifier needs each acquire released, so
every pool's core body is the same four lines. The committed tests leave those
lines out, and their RUN lines expand them with this script.

In each --split-input-file chunk, each self-loop pool that no acquire names
gets a Produce acquire and release, then a Consume acquire and release, at the
top of its tile's core, in declaration order. A tile without a core gets one
before the device's closing brace, in tile order; in a chunk with no core at
all, every ``aie.tile`` gets one.
"""

import argparse
import re
import sys
from pathlib import Path

SPLIT = re.compile(r"^// -----.*$", re.M)
FIFO = re.compile(
    r"^(\s*)aie\.objectfifo @(\w+)\((%\w+), \{(%\w+)\}, \d+ : i32\) : "
    r"!aie\.objectfifo<(memref<[^>]+>)>"
)
TILE = re.compile(r"^\s*(%\w+) = aie\.(logical_)?tile")
CORE = re.compile(r"^(\s*)(?:%\w+ = )?aie\.core\((%\w+)\) \{\s*$")
DEVICE = re.compile(r"^(\s*)aie\.device\(")
ACQUIRE = re.compile(r"aie\.objectfifo\.acquire @(\w+)\(")


class Pool:
    """A self-loop pool whose core body the chunk leaves out.

    Args:
        name: The object FIFO's symbol.
        memref: Its object type.
    """

    def __init__(self, name: str, memref: str):
        self.name = name
        self.memref = memref

    def body(self, indent: str) -> list:
        """Return the pool's filler and drainer.

        Args:
            indent: The indentation of the core's ops.

        Returns:
            The four lines, Produce before Consume.
        """
        lines = []
        for port, prefix in (("Produce", "p"), ("Consume", "c")):
            lines.append(
                f"{indent}%{prefix}_{self.name} = aie.objectfifo.acquire "
                f"@{self.name}({port}, 1) : {self.memref}"
            )
            lines.append(f"{indent}aie.objectfifo.release @{self.name}({port}, 1)")
        return lines


class Chunk:
    """One --split-input-file chunk of a lock test.

    Args:
        text: The chunk.
    """

    def __init__(self, text: str):
        self.lines = text.split("\n")

    def missing(self) -> dict:
        """Return the self-loop pools no acquire names, by tile.

        Returns:
            Each tile's pools in declaration order; with no core in the chunk,
            every ``aie.tile`` is a key, with no pools if it has none.
        """
        named = {m for line in self.lines for m in ACQUIRE.findall(line)}
        pools = {}
        if not any(CORE.match(line) for line in self.lines):
            for line in self.lines:
                tile = TILE.match(line)
                if tile and not tile.group(2):
                    pools[tile.group(1)] = []
        for line in self.lines:
            fifo = FIFO.match(line)
            if fifo and fifo.group(3) == fifo.group(4):
                if fifo.group(2) not in named:
                    pools.setdefault(fifo.group(3), []).append(
                        Pool(fifo.group(2), fifo.group(5))
                    )
        return pools

    def expand(self) -> str:
        """Return the chunk with every pool's core body written out.

        Returns:
            The expanded chunk.
        """
        pools = self.missing()
        if not any(pools.values()):
            return "\n".join(self.lines)
        out = []
        device_indent = None
        device_close = None
        for line in self.lines:
            core = CORE.match(line)
            device = DEVICE.match(line)
            if device and device_indent is None:
                device_indent = device.group(1)
            if device_close is None and line == f"{device_indent}}}":
                device_close = len(out)
            out.append(line)
            if core and core.group(2) in pools:
                for pool in pools.pop(core.group(2)):
                    out += pool.body(core.group(1) + "  ")
        tiles = [m.group(1) for line in self.lines if (m := TILE.match(line))]
        indent = device_indent + "  "
        cores = []
        for tile in sorted(pools, key=tiles.index):
            cores.append(f"{indent}aie.core({tile}) {{")
            for pool in pools[tile]:
                cores += pool.body(indent + "  ")
            cores += [f"{indent}  aie.end", f"{indent}}}"]
        return "\n".join(out[:device_close] + cores + out[device_close:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path, help="the lock test")
    args = parser.parse_args()
    chunks = SPLIT.split(args.input.read_text())
    sys.stdout.write("// -----".join(Chunk(chunk).expand() for chunk in chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
