# test_run_interleaved.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""run_interleaved on real CPU-bound callables of known relative cost, no NPU.

`work(4 * N)` costs four times `work(N)`: one Python addition per element.
"""

import argparse
import os
import subprocess
import sys
import time
from functools import partial

import pytest

from aie.utils.benchmark import cpu_wait_us, run_interleaved

N = 2_000


def work(n):
    total = 0
    for i in range(n):
        total += i
    return None


def dispatch(n, wait_s):
    """Host work, then a wait, as a dispatch and its completion take."""
    work(n)
    time.sleep(wait_s)


def test_each_callable_has_one_median_a_round():
    light, heavy = run_interleaved(
        [partial(work, N), partial(work, 4 * N)], rounds=5, calls=5
    )
    assert light.n == heavy.n == 5
    assert len(light.samples_us) == 5
    # 4 by construction; a shared host's noise is held within a factor of 2.
    assert 2 < heavy.median_us / light.median_us < 8


def test_a_group_past_the_cutoff_stops_once_settled():
    fast, slow = run_interleaved(
        [partial(work, N), partial(work, 4 * N)],
        rounds=6,
        calls=3,
        settle=2,
        cutoff=1.5,
        groups=["fast", "slow"],
    )
    assert fast.n == 6
    assert slow.n == 2


def test_a_group_stops_together_by_its_slowest():
    a_fast, a_slow, b = run_interleaved(
        [partial(work, N), partial(work, 4 * N), partial(work, 2 * N)],
        rounds=5,
        calls=3,
        settle=2,
        cutoff=1.5,
        groups=["a", "a", "b"],
    )
    assert a_fast.n == a_slow.n == 2
    assert b.n == 5


def test_a_callable_of_no_group_is_never_stopped():
    fast, slow = run_interleaved(
        [partial(work, N), partial(work, 4 * N)],
        rounds=4,
        calls=3,
        settle=2,
        cutoff=1.5,
        groups=["fast", None],
    )
    assert fast.n == slow.n == 4


def test_a_faster_figure_from_before_stops_every_group():
    (only,) = run_interleaved(
        [partial(work, N)],
        rounds=4,
        calls=3,
        settle=2,
        cutoff=1.5,
        groups=["only"],
        fastest_us=1.0,
    )
    assert only.n == 2


def test_groups_must_be_one_per_callable():
    with pytest.raises(ValueError, match="2 groups for 1 callables"):
        run_interleaved([partial(work, N)], groups=["a", "b"])


SPIN = """
import os
import sys

os.sched_setaffinity(0, {int(sys.argv[1])})
print("spinning", flush=True)
while True:
    pass
"""


def starved(cpu):
    """Time `dispatch` at nice 19 on `cpu`, and print what came of it."""
    os.sched_setaffinity(0, {cpu})
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    try:
        (s,) = run_interleaved([partial(dispatch, N, 0.0002)], rounds=3, calls=10)
    except RuntimeError as e:
        print(f"refused: {e}", flush=True)
    else:
        print(f"figure: {s.median_us:.1f} us", flush=True)


@pytest.mark.skipif(cpu_wait_us() is None, reason="no /proc/thread-self/schedstat")
def test_a_figure_whose_calls_waited_for_a_cpu_is_refused():
    # A nice-0 process on the timing thread's only core, the timing process
    # at nice 19 so that every wakeup waits for the core (measured: 3.7 to
    # 5.7 ms median); at equal nice, whether a wakeup preempts the spinner
    # depends on the scheduler's lag, and the waits come and go.
    cpu = max(os.sched_getaffinity(0))
    spinner = subprocess.Popen(
        [sys.executable, "-c", SPIN, str(cpu)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert spinner.stdout.readline().strip() == "spinning"
        timing = subprocess.Popen(
            [sys.executable, __file__, "--starved-on", str(cpu)],
            stdout=subprocess.PIPE,
            text=True,
        )
        outcome = timing.stdout.readline()
    finally:
        # The child's own exit, at nice 19, would crawl beside the spinner.
        spinner.kill()
        spinner.wait()
    assert timing.wait() == 0
    assert outcome.startswith("refused: "), outcome
    assert "for a CPU" in outcome


@pytest.mark.skipif(cpu_wait_us() is None, reason="no /proc/thread-self/schedstat")
def test_cpu_wait_only_grows():
    first = cpu_wait_us()
    work(N)
    assert cpu_wait_us() >= first >= 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--starved-on", type=int, required=True, metavar="CPU")
    starved(parser.parse_args().starved_on)
