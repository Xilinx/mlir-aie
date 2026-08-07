#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Turn an lcov coverage export into a compact, sorted markdown table for a
PR comment.

Unlike `llvm-cov report`'s raw text output, this only ever reports on the
files it's explicitly told about (via --restrict) and sorts by line coverage
ascending, so the worst-covered changed files are visible first without
scrolling through the whole codebase.
"""

import argparse
import os
import sys


def normalize_path(path):
    return os.path.normpath(path)


def parse_lcov(lcov_path, allowed):
    """Return {path: (covered_lines, total_lines)} for paths in `allowed`.

    A header can appear in multiple `SF:` blocks -- one per translation unit
    that includes it -- each covering the same set of lines. Line hits are
    unioned by line number (max hits seen for that line across all blocks)
    rather than summed across blocks, so a header's total line count isn't
    inflated by however many TUs happen to include it.
    """
    line_hits = {}
    path = None
    tracked = False
    with open(lcov_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                path = normalize_path(line[len("SF:") :])
                tracked = path in allowed
                if tracked:
                    line_hits.setdefault(path, {})
            elif tracked and line.startswith("DA:"):
                fields = line[len("DA:") :].split(",")
                if (
                    len(fields) < 2
                    or not fields[0].isdigit()
                    or not fields[1].lstrip("-").isdigit()
                ):
                    continue
                line_no = int(fields[0])
                hits = int(fields[1])
                hits_by_line = line_hits[path]
                hits_by_line[line_no] = max(hits_by_line.get(line_no, 0), hits)
            elif line == "end_of_record":
                path = None
                tracked = False
    return {
        path: (sum(1 for hits in hits_by_line.values() if hits != 0), len(hits_by_line))
        for path, hits_by_line in line_hits.items()
    }


def format_table(records, max_rows):
    # Only files with at least one executable line are meaningful to rank.
    rows = [(p, c, t) for p, (c, t) in records.items() if t > 0]
    rows.sort(key=lambda r: r[1] / r[2])

    if not rows:
        return "No coverage data for the changed files."

    total_covered = sum(c for _, c, _ in rows)
    total_lines = sum(t for _, _, t in rows)
    overall_pct = 100.0 * total_covered / total_lines

    lines = [
        f"Average line coverage across {len(rows)} changed file"
        f"{'s' if len(rows) != 1 else ''}: **{overall_pct:.1f}%** "
        f"({total_covered}/{total_lines} lines)",
        "",
        "| File | Line coverage | Lines covered/total |",
        "|---|---|---|",
    ]
    for path, covered, total_ in rows[:max_rows]:
        pct = 100.0 * covered / total_
        lines.append(f"| {path} | {pct:.1f}% | {covered}/{total_} |")
    if len(rows) > max_rows:
        remaining = len(rows) - max_rows
        lines.append("")
        lines.append(
            f"_{remaining} more file{'s' if remaining != 1 else ''} omitted "
            "for length (all with higher coverage than those shown above)._"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lcov_path", help="Path to an lcov file (see llvm-cov export -format lcov)"
    )
    parser.add_argument(
        "--restrict",
        required=True,
        help="';'-separated list of absolute file paths to report on; "
        "everything else in the lcov file is ignored",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Maximum number of files to list before truncating",
    )
    args = parser.parse_args()

    allowed = {normalize_path(p) for p in args.restrict.split(";") if p}
    try:
        records = parse_lcov(args.lcov_path, allowed)
        output = format_table(records, args.max_rows)
    except OSError as e:
        print(f"Failed to read coverage report: {e}", file=sys.stderr)
        output = "_Coverage report unavailable._"

    print(output)


if __name__ == "__main__":
    main()
