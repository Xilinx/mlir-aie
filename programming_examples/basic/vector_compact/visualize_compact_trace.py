#!/usr/bin/env python3
##===- visualize_compact_trace.py ---------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Visualize the butterfly vs scalar stream-compaction traces.
#
# Adapted from basic/event_trace/visualize_trace.py: it reuses that file's
# Chrome-tracing B/E interval parsing, but instead of one big lane timeline it
# produces a two-panel figure tuned for the kernel comparison:
#
#   (top)    timeline -- the event0->event1 region-of-interest window for each
#            kernel, drawn as a horizontal bar on its own lane (origin-shifted
#            so both windows start at t=0 and are directly comparable).
#   (bottom) bar chart -- event0->event1 core-cycle count, butterfly vs scalar,
#            each bar annotated with the cycle count and the theoretical op
#            count (~992 vector ops vs ~1024 scalar stores).
#
# Saved as compact_trace.png and (if a display/backend is available) shown
# interactively.
#
##===----------------------------------------------------------------------===##

import argparse
import json

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Theoretical op counts documented in vector_compact_kernel.cc.
_THEORY = {
    "butterfly": (992, "vector ops"),
    "scalar": (1024, "scalar stores"),
}
_COLORS = {"butterfly": "#1f77b4", "scalar": "#d62728"}


# --- parsing (kept from visualize_trace.py) --------------------------------
def parse_trace_json(trace_file):
    """Parse a Chrome-tracing JSON file and return (processes, threads, events).

    Identical structure to basic/event_trace/visualize_trace.py.
    """
    with open(trace_file, "r") as f:
        data = json.load(f)

    processes = {}
    threads = {}
    events = []
    for entry in data:
        if entry["ph"] == "M":  # Metadata
            if entry["name"] == "process_name":
                processes[entry["pid"]] = entry["args"]["name"]
            elif entry["name"] == "thread_name":
                threads[(entry["pid"], entry["tid"])] = entry["args"]["name"]
        elif entry["ph"] in ["B", "E"]:  # Begin or End events
            events.append(entry)
    return processes, threads, events


def roi_window(events):
    """Return (start_ts, end_ts) of the event0->event1 region of interest.

    Uses the first INSTR_EVENT_0 begin and the first INSTR_EVENT_1 begin, the
    same markers get_cycles() uses for the cycle delta.
    """
    start = end = None
    for e in events:
        if e["ph"] != "B":
            continue
        if e["name"] == "INSTR_EVENT_0" and start is None:
            start = e["ts"]
        elif e["name"] == "INSTR_EVENT_1" and end is None:
            end = e["ts"]
    return start, end


def build_intervals(events):
    """Pair B/E events into intervals (kept from visualize_trace.py)."""
    active = {}
    intervals = []
    for event in events:
        key = (event["pid"], event["tid"], event["name"])
        if event["ph"] == "B":
            active[key] = event["ts"]
        elif event["ph"] == "E" and key in active:
            start_ts = active.pop(key)
            intervals.append(
                {
                    "name": event["name"],
                    "start": start_ts,
                    "end": event["ts"],
                    "duration": event["ts"] - start_ts,
                }
            )
    return intervals


# --- plotting --------------------------------------------------------------
def plot_compare(
    bfly_json,
    scal_json,
    bfly_cycles,
    scal_cycles,
    survivors,
    output_file,
    show,
):
    panels = []  # (label, roi_cycles, vector_frac)
    for label, jpath, cyc in (
        ("butterfly", bfly_json, bfly_cycles),
        ("scalar", scal_json, scal_cycles),
    ):
        _, _, events = parse_trace_json(jpath)
        start, end = roi_window(events)
        roi = (end - start) if (start is not None and end is not None) else cyc

        # Fraction of the ROI the INSTR_VECTOR unit was active (informative for
        # the butterfly; ~0 for the scalar baseline).
        intervals = build_intervals(events)
        vec_cycles = 0
        if start is not None and end is not None:
            for it in intervals:
                if it["name"] == "INSTR_VECTOR" and start <= it["start"] <= end:
                    vec_cycles += it["duration"]
        vec_frac = (vec_cycles / roi) if roi else 0.0
        panels.append((label, roi if roi else cyc, vec_frac))

    fig, (ax_time, ax_bar) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.4]}
    )

    # ---- (top) timeline: origin-shifted ROI window per kernel --------------
    for lane, (label, roi, _vf) in enumerate(panels):
        ax_time.add_patch(
            mpatches.Rectangle(
                (0, lane - 0.35),
                roi,
                0.7,
                facecolor=_COLORS[label],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
            )
        )
        ax_time.text(
            roi / 2,
            lane,
            f"{label}: {roi} cycles (event0→event1)",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            weight="bold",
        )
    max_roi = max(p[1] for p in panels) if panels else 1
    ax_time.set_xlim(0, max_roi * 1.05)
    ax_time.set_ylim(-0.6, len(panels) - 0.4)
    ax_time.set_yticks(range(len(panels)))
    ax_time.set_yticklabels([p[0] for p in panels])
    ax_time.set_xlabel("AIE core cycles (origin-shifted to event0)")
    ax_time.set_title(
        f"Stream-compaction region-of-interest (survivors={survivors}/1024, bf16)",
        weight="bold",
    )
    ax_time.grid(True, axis="x", alpha=0.3)

    # ---- (bottom) cycle-count comparison bar chart -------------------------
    labels = [p[0] for p in panels]
    cycles = [p[1] for p in panels]
    colors = [_COLORS[l] for l in labels]
    bars = ax_bar.bar(labels, cycles, color=colors, edgecolor="black", alpha=0.85)

    ymax = max(cycles) if cycles else 1
    for bar, (label, roi, vf) in zip(bars, panels):
        theory_n, theory_unit = _THEORY[label]
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{roi} cycles\n(~{theory_n} {theory_unit})",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    speedup = (scal_cycles / bfly_cycles) if bfly_cycles else float("nan")
    ax_bar.set_ylim(0, ymax * 1.22)
    ax_bar.set_ylabel("event0→event1 core cycles")
    ax_bar.set_title(
        f"Cycle count: butterfly vs scalar  (speedup {speedup:.1f}x)",
        weight="bold",
    )
    ax_bar.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to: {output_file}")

    print("\nSummary:")
    for label, roi, vf in panels:
        theory_n, theory_unit = _THEORY[label]
        print(
            f"  {label:<10}: {roi:>6} cycles  "
            f"(~{theory_n} {theory_unit}, vector-active {vf*100:.0f}% of ROI)"
        )
    print(f"  speedup   : {speedup:.1f}x")

    if show:
        try:
            plt.show()
        except Exception as e:  # headless / no display backend
            print(f"(interactive show skipped: {e})")


def main():
    p = argparse.ArgumentParser(
        description="Visualize butterfly vs scalar stream-compaction traces"
    )
    p.add_argument("--butterfly-json", required=True)
    p.add_argument("--scalar-json", required=True)
    p.add_argument("--butterfly-cycles", type=float, required=True)
    p.add_argument("--scalar-cycles", type=float, required=True)
    p.add_argument("--survivors", type=int, default=0)
    p.add_argument("--output", default="compact_trace.png")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="do not attempt an interactive plt.show() (e.g. headless CI)",
    )
    args = p.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    plot_compare(
        args.butterfly_json,
        args.scalar_json,
        int(args.butterfly_cycles)
        if float(args.butterfly_cycles).is_integer()
        else args.butterfly_cycles,
        int(args.scalar_cycles)
        if float(args.scalar_cycles).is_integer()
        else args.scalar_cycles,
        args.survivors,
        args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
