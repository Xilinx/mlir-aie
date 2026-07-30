# coherence.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Where each byte of an allocation currently lives, and at what granularity.

Split out of :mod:`tensor_class` so the residency bookkeeping can be read,
and tested, without the tensor surface layered over it.
"""

import bisect
import os

# Host<->device synchronization is not byte-granular: on a part that is not
# cache-coherent the runtime maintains whole cache lines (it walks
# ``sysconf(_SC_LEVEL1_DCACHE_LINESIZE)``-sized steps and the CPU acts on the
# entire line containing an address), so a sync for one region unavoidably acts
# on every byte sharing its first and last line. Sub-region views must therefore
# not share a line, or one view's sync can write a neighbor's stale host copy
# over data the device just produced. Lines, not pages: pages are the unit of
# address translation and protection, lines are the unit of coherence.
_SYSFS_CACHE_DIR = "/sys/devices/system/cpu/cpu0/cache"


def _read_sysfs_line_size(cache_dir=_SYSFS_CACHE_DIR):
    """Line size of the first level-1 *data* cache, or None.

    Cache indices are not ordered by level or type, so select on both rather
    than assuming index0 is L1d: on a machine that enumerates the instruction
    cache first, that assumption reads the wrong line size.
    """
    for index in range(8):
        base = f"{cache_dir}/index{index}"
        try:
            with open(f"{base}/level") as fp:
                level = int(fp.read().strip())
            with open(f"{base}/type") as fp:
                kind = fp.read().strip()
            if level != 1 or kind not in ("Data", "Unified"):
                continue
            with open(f"{base}/coherency_line_size") as fp:
                return int(fp.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _detect_coherence_granule(default=64):
    """Largest plausible coherence granule for this host, never below ``default``.

    ``sysconf`` first because it is the portable spelling and is what the
    runtime's own flush loop uses; the sysfs scan is a fallback for platforms
    that do not report it. Rounded up to a power of two so the value can be used
    as an alignment, and floored so a missing or nonsensical report can only ever
    make the check stricter, never weaker. Absent both sources (notably Windows,
    where ``os.sysconf`` does not exist) the floor is what is used, which is the
    line size of every architecture this runs on today.
    """
    reported = None
    try:
        reported = os.sysconf("SC_LEVEL1_DCACHE_LINESIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if not reported or reported < 0:
        reported = _read_sysfs_line_size()
    granule = max(reported or 0, default)
    # Round up to a power of two: alignment arithmetic assumes it, and a cache
    # line is one on every architecture, but nothing in the reporting path
    # promises it.
    return 1 << (granule - 1).bit_length()


COHERENCE_GRANULE = _detect_coherence_granule()


class _CoherenceMap:
    """Which byte ranges of one allocation the host has written, and which the
    device has.

    The state belongs to the memory, not to whichever tensor happens to name it.
    Ranges are kept coalesced, so an arena whose windows are all in the same
    state costs one entry, and a whole-buffer reconcile is one operation rather
    than one per view.
    """

    HOST = "cpu"
    DEVICE = "npu"

    def __init__(self, nbytes, state, granule=None):
        self._spans = [[0, nbytes, state]]
        # Where each span ends, kept beside the spans purely so a lookup can
        # bisect plain integers. Bisecting the spans themselves needs a key
        # function, which is a Python call at every step of the search.
        self._ends = [nbytes]
        self._granule = COHERENCE_GRANULE if granule is None else granule
        # The whole allocation in one state, or None when it is mixed. Kept as a
        # field because it answers most queries outright, and the queries are on
        # the path a dispatch takes for every argument.
        self.uniform = state

    def set(self, start, end, state):
        """Record ``[start, end)`` as ``state``.

        Only the spans the range actually touches are rewritten. Rebuilding the
        whole list instead is quadratic over a buffer written a window at a
        time, which is the case this exists to serve.
        """
        if start >= end:
            return
        spans = self._spans
        # First span ending after start, and first span beginning at or after
        # end: everything between them is what this write disturbs.
        ends = self._ends
        # ends[i] is where span i stops, so the first span this write disturbs
        # is the first one stopping after start, and the last is the one holding
        # end - 1.
        first = bisect.bisect_right(ends, start)
        last = bisect.bisect_left(ends, end) + 1
        replacement = []
        if first < last:
            head = spans[first]
            if head[0] < start:  # keep the part of the first span before us
                replacement.append([head[0], start, head[2]])
            replacement.append([start, end, state])
            tail = spans[last - 1]
            if tail[1] > end:  # and the part of the last one after us
                replacement.append([end, tail[1], tail[2]])
        else:
            replacement.append([start, end, state])
        # Merge with the neighbours on either side, so a buffer written window
        # by window collapses back to one span once the windows meet.
        while first > 0 and spans[first - 1][2] == replacement[0][2]:
            replacement[0][0] = spans[first - 1][0]
            first -= 1
        while last < len(spans) and spans[last][2] == replacement[-1][2]:
            replacement[-1][1] = spans[last][1]
            last += 1
        merged = [replacement[0]]
        for span in replacement[1:]:
            if merged[-1][2] == span[2]:
                merged[-1][1] = span[1]
            else:
                merged.append(span)
        spans[first:last] = merged
        ends[first:last] = [span[1] for span in merged]
        self._check_granule(merged, start, end, state)
        self.uniform = spans[0][2] if len(spans) == 1 else None

    def _check_granule(self, merged, start, end, state):
        """No coherence granule may hold regions in two different states.

        This is what makes the map mean anything: reconciling one range acts on
        whole cache lines, so two states sharing a line cannot be maintained
        independently and one of them will be written over the other.

        Only the boundaries this write introduces are checked. Every other
        boundary was checked when it was made and nothing here moves it, so the
        property is carried forward rather than re-established, which keeps a
        buffer written window by window from costing more with each window. The
        ends of the allocation are not boundaries: nothing lies beyond them.
        """
        granule = self._granule
        # Every edge of the rewritten block: where it meets its neighbours, and
        # the boundaries inside it. At most a handful, whatever the buffer holds.
        edges = {merged[0][0]}
        edges.update(span[1] for span in merged)
        end_of_buffer = self._spans[-1][1]
        for edge in edges:
            if edge and edge != end_of_buffer and edge % granule:
                raise ValueError(
                    f"recording {state!r} for bytes [{start}, {end}) would leave a "
                    f"state boundary at byte {edge}, which is not on a {granule}-byte "
                    f"coherence granule. Host and device are reconciled a cache line "
                    f"at a time, so the regions either side of that boundary could "
                    f"not be transferred independently."
                )

    def get(self, start, end):
        """The state of ``[start, end)``, or None if it is not uniform."""
        if self.uniform is not None:
            return self.uniform
        seen = {v for lo, hi, v in self._spans if lo < end and hi > start}
        return seen.pop() if len(seen) == 1 else None

    def any_host_written(self, start, end):
        if self.uniform is not None:
            return self.uniform == self.HOST
        return any(
            v == self.HOST for lo, hi, v in self._spans if lo < end and hi > start
        )

    def ranges(self, start, end, state):
        """The sub-ranges of ``[start, end)`` currently in ``state``."""
        return [
            (max(lo, start), min(hi, end))
            for lo, hi, value in self._spans
            if value == state and lo < end and hi > start
        ]
