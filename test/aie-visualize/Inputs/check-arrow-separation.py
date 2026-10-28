# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import sys
from fractions import Fraction


MINIMUM_GAP = Fraction("1.2")


def axis_runs(path):
    text = open(path, encoding="utf-8").read()
    runs = []
    for edge, match in enumerate(re.finditer(
        r'^\s*(\w+) -> (\w+) \[(.*)\];$', text, re.MULTILINE
    )):
        attributes = match.group(3)
        if 'class="legend' in attributes:
            continue
        position = re.search(r'pos="([^"]+)"', attributes)
        if not position:
            continue
        owner = f"{match.group(1)} -> {match.group(2)} #{edge}"
        endpoints = frozenset((match.group(1), match.group(2)))
        flow = re.search(r'\bflow src-(\d+)\b', attributes)
        flow_owner = flow.group(1) if flow else owner
        fields = [
            field
            for field in position.group(1).split()
            if not field.startswith(("e,", "s,"))
        ]
        points = [tuple(map(Fraction, field.split(","))) for field in fields]
        knots = points[::3] if len(points) >= 4 else points
        for first, second in zip(knots, knots[1:]):
            if first[1] == second[1]:
                runs.append(
                    (
                        "h",
                        first[1],
                        *sorted((first[0], second[0])),
                        flow_owner,
                        endpoints,
                        owner,
                    )
                )
            elif first[0] == second[0]:
                runs.append(
                    (
                        "v",
                        first[0],
                        *sorted((first[1], second[1])),
                        flow_owner,
                        endpoints,
                        owner,
                    )
                )
    return runs


def main():
    runs = axis_runs(sys.argv[1])
    collinear_overlaps = []
    narrow_gaps = []
    crossings = 0
    for index, first in enumerate(runs):
        (
            first_axis,
            first_cross,
            first_low,
            first_high,
            first_flow,
            first_endpoints,
            first_owner,
        ) = first
        for second in runs[index + 1 :]:
            (
                second_axis,
                second_cross,
                second_low,
                second_high,
                second_flow,
                second_endpoints,
                second_owner,
            ) = second
            if first_flow == second_flow or first_endpoints & second_endpoints:
                continue
            if first_axis != second_axis:
                horizontal = first if first_axis == "h" else second
                vertical = second if first_axis == "h" else first
                if (
                    horizontal[2] < vertical[1] < horizontal[3]
                    and vertical[2] < horizontal[1] < vertical[3]
                ):
                    crossings += 1
                continue
            overlap = min(first_high, second_high) - max(first_low, second_low)
            if overlap <= 0:
                continue
            gap = abs(first_cross - second_cross)
            conflict = (first_owner, second_owner, overlap, gap)
            if gap == 0:
                collinear_overlaps.append(conflict)
            elif gap < MINIMUM_GAP:
                narrow_gaps.append(conflict)
    print(
        f"collinear_overlaps={len(collinear_overlaps)} "
        f"narrow_gaps={len(narrow_gaps)} crossings={crossings}"
    )
    for first, second, overlap, gap in collinear_overlaps + narrow_gaps:
        print(
            f"{first} | {second}: overlap={float(overlap):.2f} "
            f"gap={float(gap):.2f}"
        )
    if collinear_overlaps or narrow_gaps:
        raise RuntimeError("found overlapping or insufficiently separated arrow runs")


if __name__ == "__main__":
    main()