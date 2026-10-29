# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import sys
from fractions import Fraction


MINIMUM_GAP = Fraction("1.2")


def is_straight(first, control1, control2, second):
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    return all(
        (point[0] - first[0]) * dy == (point[1] - first[1]) * dx
        for point in (control1, control2)
    )


def axis_runs(path):
    text = open(path, encoding="utf-8").read()
    runs = []
    diagonals = []
    missing_geometry = []
    for edge, match in enumerate(re.finditer(
        r'^\s*(\w+) -> (\w+) \[(.*)\];$', text, re.MULTILINE
    )):
        attributes = match.group(3)
        if 'class="legend' in attributes:
            continue
        position = re.search(r'pos="([^"]+)"', attributes)
        if not position:
            missing_geometry.append(f"{match.group(1)} -> {match.group(2)} #{edge}")
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
        segments = list(zip(knots, knots[1:]))
        for index, (first, second) in enumerate(segments):
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
            else:
                if len(points) < 4 or is_straight(
                    first, points[3 * index + 1], points[3 * index + 2], second
                ):
                    diagonals.append((owner, first, second))
    return runs, diagonals, missing_geometry


def main():
    runs, diagonals, missing_geometry = axis_runs(sys.argv[1])
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
        f"missing_geometry={len(missing_geometry)} "
        f"collinear_overlaps={len(collinear_overlaps)} "
        f"narrow_gaps={len(narrow_gaps)} diagonals={len(diagonals)} "
        f"crossings={crossings}"
    )
    for first, second, overlap, gap in collinear_overlaps + narrow_gaps:
        print(
            f"{first} | {second}: overlap={float(overlap):.2f} "
            f"gap={float(gap):.2f}"
        )
    for owner, first, second in diagonals:
        print(f"{owner}: diagonal={first} to {second}")
    for owner in missing_geometry:
        print(f"{owner}: missing pos")
    if missing_geometry or collinear_overlaps or narrow_gaps or diagonals:
        raise RuntimeError("found invalid arrow runs")


if __name__ == "__main__":
    main()