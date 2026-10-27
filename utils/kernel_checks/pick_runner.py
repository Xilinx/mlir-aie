#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Pick the bench runner a nightly's NPU leg runs on: its primary, or the backup.

Each NPU generation has two bench runners: the one labelled
``nightly-primary`` runs every nightly, so a series of numbers comes from
one machine, and the one labelled ``nightly-backup`` runs it only when the
primary is down. GitHub queues a job for an offline runner for a day rather
than failing it, so a job can't find out by itself: this asks.

Per runner type, it dispatches ``benchRunnerProbe.yml``, a job that does
nothing on that type's primary, and waits for a runner to take it. Taken
within ``--wait-minutes``: the primary is up, and the leg runs there. Not
taken: the probe is cancelled, so it does not run later, and the leg runs
on the backup. The wait is long enough for the primary to finish a job it
is busy with (a merge's hardware tests, the other nightly), so only a
machine that is down sends the leg elsewhere.

Writes ``runs_on`` to ``$GITHUB_OUTPUT``: per runner type, the labels the
leg's ``runs-on`` takes, and a line per type to the job summary.

    GITHUB_TOKEN=... python3 utils/kernel_checks/pick_runner.py \
        --runner-types aie2-4col aie2p-8col --run-id 123 --ref main
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
PROBE = "benchRunnerProbe.yml"
LABELS = {"primary": "nightly-primary", "backup": "nightly-backup"}
# A probe job a runner has taken.
TAKEN = ("in_progress", "completed")


class ApiError(Exception):
    pass


def github(token: str, timeout: float = 30, api: str = API):
    """Return a function that sends one API request and returns its JSON.

    Args:
        token: The token to send, or empty for none.
        timeout: Seconds to wait for each answer.
        api: The API's base URL.

    Returns:
        ``call(method, path, body=None)``: the answer's JSON, or None for an
        empty answer.
    """

    def call(method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{api}{path}", data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                raw = res.read()
        except urllib.error.HTTPError as e:
            raise ApiError(f"HTTP {e.code} for {method} {path}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ApiError(f"{e} for {method} {path}") from e
        return json.loads(raw) if raw else None

    return call


def probe_title(runner_type: str, run_id: str) -> str:
    """Name a probe run as ``benchRunnerProbe.yml``'s ``run-name`` writes it."""
    return f"Probe {runner_type} for run {run_id}"


def verdict(jobs: list, waited: float, wait: float) -> str | None:
    """Whether a type's primary is up, from its probe's jobs.

    Args:
        jobs: The probe run's jobs, as the API lists them; empty while the
            run has none yet.
        waited: Minutes since the probe was dispatched.
        wait: Minutes to wait for a runner to take the probe.

    Returns:
        ``"primary"`` once a runner has taken the probe, ``"backup"`` when
        none has within the wait, None while it is still worth waiting.
    """
    if any(j.get("status") in TAKEN and j.get("runner_name") for j in jobs):
        return "primary"
    return "backup" if waited >= wait else None


def find_probe(call, repo: str, title: str, since: str) -> dict | None:
    """Find the probe run named `title` dispatched since `since`, or None."""
    query = urllib.parse.urlencode(
        {"event": "workflow_dispatch", "created": f">={since}", "per_page": 50}
    )
    runs = call("GET", f"/repos/{repo}/actions/workflows/{PROBE}/runs?{query}")[
        "workflow_runs"
    ]
    return next((r for r in runs if r.get("display_title") == title), None)


def pick(
    call, repo: str, ref: str, run_id: str, runner_types: list, wait: float, poll: float
) -> dict:
    """Probe each type's primary and decide where each leg runs.

    Args:
        call: A function from github().
        repo: ``owner/name``.
        ref: The branch the probes run from.
        run_id: The nightly's run, named in each probe's title.
        runner_types: The runner types to probe.
        wait: Minutes to wait for a runner to take a probe.
        poll: Seconds between looks.

    Returns:
        Per runner type: {role, runner, note}.
    """
    since = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=2)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = time.monotonic()
    for t in runner_types:
        call(
            "POST",
            f"/repos/{repo}/actions/workflows/{PROBE}/dispatches",
            {"ref": ref, "inputs": {"runner_type": t, "for_run": str(run_id)}},
        )
    picked = {}
    while len(picked) < len(runner_types):
        time.sleep(poll)
        waited = (time.monotonic() - start) / 60
        for t in runner_types:
            if t in picked:
                continue
            probe = find_probe(call, repo, probe_title(t, run_id), since)
            jobs = (
                call("GET", f"/repos/{repo}/actions/runs/{probe['id']}/jobs")["jobs"]
                if probe
                else []
            )
            role = verdict(jobs, waited, wait)
            if role == "primary":
                runner = next(j["runner_name"] for j in jobs if j.get("runner_name"))
                picked[t] = {
                    "role": role,
                    "runner": runner,
                    "note": f"{runner} took the probe after {waited:.0f} min",
                }
            elif role == "backup":
                # So the probe does not take the primary later, when it is back.
                if probe:
                    call("POST", f"/repos/{repo}/actions/runs/{probe['id']}/cancel")
                picked[t] = {
                    "role": role,
                    "runner": None,
                    "note": f"no runner took the probe in {wait:.0f} min",
                }
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pick the bench runner each NPU leg of a nightly runs on."
    )
    parser.add_argument(
        "--runner-types", nargs="+", required=True, choices=["aie2-4col", "aie2p-8col"]
    )
    parser.add_argument(
        "--run-id", required=True, help="the nightly's run id, named in each probe"
    )
    parser.add_argument(
        "--ref", default="main", help="the branch the probe workflow runs from"
    )
    parser.add_argument(
        "--role",
        choices=["auto", "primary", "backup"],
        default="auto",
        help="auto probes the primaries; primary or backup picks without asking",
    )
    parser.add_argument("--wait-minutes", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    repo = os.environ.get("GITHUB_REPOSITORY", "Xilinx/mlir-aie")
    if args.role == "auto":
        call = github(os.environ.get("GITHUB_TOKEN", ""))
        try:
            picked = pick(
                call,
                repo,
                args.ref,
                args.run_id,
                args.runner_types,
                args.wait_minutes,
                args.poll_seconds,
            )
        except ApiError as e:
            # GitHub not answering is no reason to skip the night: the primary
            # it is, and the job waits for it as it always could.
            print(
                f"::warning::could not probe the bench runners ({e}): running on the primaries"
            )
            picked = {
                t: {"role": "primary", "runner": None, "note": f"not probed: {e}"}
                for t in args.runner_types
            }
    else:
        picked = {
            t: {"role": args.role, "runner": None, "note": "chosen by the dispatch"}
            for t in args.runner_types
        }
    out = {t: ["self-hosted", "bench", t, LABELS[p["role"]]] for t, p in picked.items()}
    lines = [f"- {t}: {p['role']} ({p['note']})" for t, p in picked.items()]
    print("\n".join(lines))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"runs_on={json.dumps(out)}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("Bench runners:\n" + "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
