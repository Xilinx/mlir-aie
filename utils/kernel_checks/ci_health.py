#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record CI health for the maintainer dashboard.

Asks GitHub's API, with the workflow's own token, for three things and
writes them to one file, ``status.json``, which publishCiHealth.yml installs
as ``gh-pages:ci-health/status.json``; the dashboard reads it from there, so
no browser asks GitHub itself.

* **The critical test workflows** (``critical`` in ``ci_workflows.json``:
  those that gate a merge). Per workflow: its latest finished run on main
  (pushes and the nightly alike) and, when it has ``legs``, that run's
  jobs; and over the last ``WINDOW_DAYS`` days how often it passed on main,
  on pull requests and in the merge queue, how many commits needed a re-run
  to pass (flaky), how long a run took and waited to start, and, from its
  latest few runs, its slowest jobs.
* **Merge time**: the pull requests merged in the window, how long each was
  open, and how many are open now and for how long.
* **The other scheduled jobs** (``workflows``): each one's recent scheduled
  runs on main and, for one marked ``legs``, the jobs of the latest, so a leg
  that failed or never found a runner can be named. A workflow is in one
  list or the other, never both.

It also keeps ``history.json``: one point a day of the critical workflows'
and merge time's figures, so the page can chart them. Pass the published
one with ``--history``; today's point is replaced as the day goes on.

    GITHUB_TOKEN=... python3 utils/kernel_checks/ci_health.py --out ci-health \\
        [--history published/history.json]

Only what the page reads is kept. Whatever GitHub will not answer is
recorded with its error; the rest still publishes.
"""

import argparse
import datetime
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKFLOWS = HERE / "ci_workflows.json"
API = "https://api.github.com"
SCHEMA = 1
RUNS_KEPT = 12
WINDOW_DAYS = 14
HISTORY_DAYS = 180
# Pages of 100 runs read per critical workflow; the window rarely needs more.
RUN_PAGES = 10
# Latest completed runs per critical workflow whose jobs are timed.
JOB_SAMPLE = 6
SLOWEST_JOBS = 6
# Pages of 100 pull requests read, newest updated first.
PULL_PAGES = 5
# A conclusion that counts against a pass rate; cancelled and skipped do not.
FAILED = ("failure", "timed_out", "startup_failure")

RUN_FIELDS = {
    "id": "id",
    "url": "html_url",
    "status": "status",
    "conclusion": "conclusion",
    "created_at": "created_at",
    "run_started_at": "run_started_at",
    "head_sha": "head_sha",
    "title": "display_title",
    "attempt": "run_attempt",
}
JOB_FIELDS = {
    "name": "name",
    "url": "html_url",
    "status": "status",
    "conclusion": "conclusion",
    "created_at": "created_at",
    "started_at": "started_at",
}


def load_workflows(path: Path = WORKFLOWS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(item: dict, fields: dict) -> dict:
    return {ours: item.get(theirs) for ours, theirs in fields.items()}


class ApiError(Exception):
    pass


def github(token: str, timeout: float = 30):
    """A function GETting one API path as JSON, with the token if any."""

    def get(path: str):
        req = urllib.request.Request(f"{API}{path}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.load(res)
        except urllib.error.HTTPError as e:
            raise ApiError(f"HTTP {e.code} for {path}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ApiError(f"{e} for {path}") from e

    return get


def when(stamp):
    """An API timestamp as an aware datetime, or None."""
    if not stamp:
        return None
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def minutes(a, b):
    a, b = when(a), when(b)
    return None if a is None or b is None else (b - a).total_seconds() / 60


def percentile(values, q):
    """The nearest-rank ``q`` percentile of ``values``, or None."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    rank = max(1, min(len(xs), math.ceil(q * len(xs))))
    return xs[rank - 1]


def rounded(x, digits=1):
    return None if x is None else round(x, digits)


def pass_rate(runs):
    """{passed, failed, rate}: completed runs that passed, failed, and the share."""
    passed = sum(r["conclusion"] == "success" for r in runs)
    failed = sum(r["conclusion"] in FAILED for r in runs)
    return {
        "passed": passed,
        "failed": failed,
        "rate": rounded(passed / (passed + failed), 3) if passed + failed else None,
    }


# ----------------------------------------------------------------------
# The scheduled workflows
# ----------------------------------------------------------------------


def scheduled(get, repo: str, workflows: list) -> list:
    """Each workflow's recent scheduled runs on main, and its latest run's
    jobs when it has legs."""
    out = []
    for wf in workflows:
        entry = dict(wf)
        try:
            runs = get(
                f"/repos/{repo}/actions/workflows/{wf['file']}/runs"
                f"?branch=main&event=schedule&per_page={RUNS_KEPT}"
            )
            entry["runs"] = [pick(r, RUN_FIELDS) for r in runs.get("workflow_runs", [])]
            entry["jobs"] = None
            if wf.get("legs") and entry["runs"]:
                jobs = get(
                    f"/repos/{repo}/actions/runs/{entry['runs'][0]['id']}/jobs"
                    "?per_page=100"
                )
                entry["jobs"] = [pick(j, JOB_FIELDS) for j in jobs.get("jobs", [])]
        except ApiError as e:
            entry.update(runs=None, jobs=None, error=str(e))
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# The critical test workflows
# ----------------------------------------------------------------------


def window_runs(get, repo: str, file: str, since: datetime.datetime) -> list:
    """Every run of ``file`` created since ``since``, any event, newest first."""
    created = urllib.parse.quote(f">={since.date().isoformat()}")
    runs = []
    for page in range(1, RUN_PAGES + 1):
        got = get(
            f"/repos/{repo}/actions/workflows/{file}/runs"
            f"?created={created}&per_page=100&page={page}"
        ).get("workflow_runs", [])
        runs.extend(got)
        if len(got) < 100:
            break
    return [r for r in runs if (when(r.get("created_at")) or since) >= since]


def flaky(runs: list) -> dict:
    """Commits whose checks failed and then passed: a run that passed on a
    later attempt, or a commit with both a failed and a passed run.
    {commits, flaky, rate} over the pull-request and merge-queue commits
    that finished."""
    by_sha = {}
    for r in runs:
        if r.get("event") not in ("pull_request", "merge_group"):
            continue
        if r.get("status") != "completed":
            continue
        by_sha.setdefault(r["head_sha"], []).append(r)
    shaky = 0
    for rs in by_sha.values():
        conclusions = {r["conclusion"] for r in rs}
        retried = any(r["conclusion"] == "success" and (r.get("run_attempt") or 1) > 1 for r in rs)
        if retried or ("success" in conclusions and conclusions & set(FAILED)):
            shaky += 1
    n = len(by_sha)
    return {"commits": n, "flaky": shaky, "rate": rounded(shaky / n, 3) if n else None}


def slowest_jobs(jobs_by_run: list) -> list:
    """Per job name over the sampled runs: median and 90th percentile
    minutes, median minutes queued, how many failed; slowest first."""
    by_name = {}
    for jobs in jobs_by_run:
        for j in jobs:
            if j.get("status") != "completed" or j.get("conclusion") == "skipped":
                continue
            s = by_name.setdefault(j["name"], {"took": [], "queued": [], "failed": 0, "n": 0})
            s["n"] += 1
            s["took"].append(minutes(j.get("started_at"), j.get("completed_at")))
            s["queued"].append(minutes(j.get("created_at"), j.get("started_at")))
            s["failed"] += j.get("conclusion") in FAILED
    rows = [
        {
            "name": name,
            "n": s["n"],
            "minutes_median": rounded(percentile(s["took"], 0.5)),
            "minutes_p90": rounded(percentile(s["took"], 0.9)),
            "queued_median": rounded(percentile(s["queued"], 0.5)),
            "failed": s["failed"],
        }
        for name, s in by_name.items()
    ]
    rows.sort(key=lambda r: -(r["minutes_median"] or 0))
    return rows[:SLOWEST_JOBS]


def critical(get, repo: str, workflows: list, now: datetime.datetime) -> list:
    """Per critical workflow, its figures over the window."""
    since = now - datetime.timedelta(days=WINDOW_DAYS)
    out = []
    for wf in workflows:
        entry = dict(wf)
        try:
            runs = window_runs(get, repo, wf["file"], since)
            done = [r for r in runs if r.get("status") == "completed"]
            on_main = [r for r in done if r.get("head_branch") == "main" and r.get("event") in ("push", "schedule")]
            prs = [r for r in done if r.get("event") == "pull_request"]
            queue = [r for r in done if r.get("event") == "merge_group"]
            passed = [r for r in done if r.get("conclusion") == "success"]
            took = [minutes(r.get("run_started_at"), r.get("updated_at")) for r in passed]
            waited = [minutes(r.get("created_at"), r.get("run_started_at")) for r in done]
            latest_main = on_main[0] if on_main else None
            # Its latest run on main may still be going: judged by the newest
            # one that is, which the page compares with `every` for lateness.
            newest_main = next((r for r in runs if r.get("head_branch") == "main" and r.get("event") in ("push", "schedule")), None)
            sample = [r for r in done if r.get("event") in ("pull_request", "merge_group", "push")][:JOB_SAMPLE]
            jobs = []
            for r in sample:
                got = get(f"/repos/{repo}/actions/runs/{r['id']}/jobs?per_page=100").get("jobs", [])
                jobs.append([{**pick(j, JOB_FIELDS), "completed_at": j.get("completed_at")} for j in got])
            main_jobs = None
            if wf.get("legs") and latest_main:
                got = get(f"/repos/{repo}/actions/runs/{latest_main['id']}/jobs?per_page=100").get("jobs", [])
                main_jobs = [pick(j, JOB_FIELDS) for j in got]
            entry.update(
                runs=len(runs),
                main=pass_rate(on_main),
                latest_main=pick(latest_main, RUN_FIELDS) if latest_main else None,
                newest_main=pick(newest_main, RUN_FIELDS) if newest_main else None,
                latest_main_jobs=main_jobs,
                pull_requests=pass_rate(prs),
                merge_queue=pass_rate(queue),
                flaky=flaky(runs),
                minutes_median=rounded(percentile(took, 0.5)),
                minutes_p90=rounded(percentile(took, 0.9)),
                queued_median=rounded(percentile(waited, 0.5)),
                slowest_jobs=slowest_jobs(jobs),
                jobs_sampled=len(sample),
            )
        except ApiError as e:
            entry["error"] = str(e)
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# Merge time
# ----------------------------------------------------------------------


def merge_time(get, repo: str, now: datetime.datetime) -> dict:
    """Pull requests merged in the window, how long each was open, and how
    many are open now and for how long."""
    since = now - datetime.timedelta(days=WINDOW_DAYS)
    try:
        merged = []
        for page in range(1, PULL_PAGES + 1):
            got = get(
                f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc"
                f"&per_page=100&page={page}"
            )
            for p in got:
                m = when(p.get("merged_at"))
                if m and m >= since:
                    merged.append((m - when(p["created_at"])).total_seconds() / 3600)
            if len(got) < 100 or when(got[-1]["updated_at"]) < since:
                break
        open_ages, drafts = [], 0
        for page in range(1, PULL_PAGES + 1):
            got = get(f"/repos/{repo}/pulls?state=open&per_page=100&page={page}")
            for p in got:
                open_ages.append((now - when(p["created_at"])).total_seconds() / 86400)
                drafts += bool(p.get("draft"))
            if len(got) < 100:
                break
    except ApiError as e:
        return {"error": str(e)}
    return {
        "merged": len(merged),
        "hours_median": rounded(percentile(merged, 0.5)),
        "hours_p90": rounded(percentile(merged, 0.9)),
        "open": len(open_ages),
        "open_drafts": drafts,
        "open_over_30_days": sum(a > 30 for a in open_ages),
    }


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------


def history_point(status: dict) -> dict:
    """Today's figures, as one point of history.json."""
    m = status.get("merge_time") or {}
    return {
        "date": status["generated_at"][:10],
        "merge": {k: m.get(k) for k in ("merged", "hours_median", "hours_p90", "open")},
        "critical": {
            w["file"]: {
                "main": (w.get("main") or {}).get("rate"),
                "pull_requests": (w.get("pull_requests") or {}).get("rate"),
                "flaky": (w.get("flaky") or {}).get("rate"),
                "minutes_median": w.get("minutes_median"),
                "queued_median": w.get("queued_median"),
            }
            for w in status.get("critical", [])
            if not w.get("error")
        },
    }


def update_history(old, point: dict, now: datetime.datetime) -> dict:
    """``old`` with today's point replaced or added, and points older than
    ``HISTORY_DAYS`` dropped."""
    points = [p for p in ((old or {}).get("points") or []) if p.get("date") != point["date"]]
    points.append(point)
    oldest = (now - datetime.timedelta(days=HISTORY_DAYS)).date().isoformat()
    points = sorted((p for p in points if p["date"] >= oldest), key=lambda p: p["date"])
    return {"schema": SCHEMA, "window_days": WINDOW_DAYS, "points": points}


def collect(get, repo: str, config: dict, now: datetime.datetime) -> dict:
    """The status file's contents."""
    return {
        "schema": SCHEMA,
        "generated_at": now.astimezone(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repo": repo,
        "window_days": WINDOW_DAYS,
        "critical": critical(get, repo, config.get("critical", []), now),
        "merge_time": merge_time(get, repo, now),
        "workflows": scheduled(get, repo, config.get("workflows", [])),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", required=True, type=Path, help="directory for status.json and history.json")
    parser.add_argument("--history", type=Path, help="the published history.json, if any")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "Xilinx/mlir-aie"))
    parser.add_argument("--workflows", type=Path, default=WORKFLOWS)
    args = parser.parse_args(argv)

    now = datetime.datetime.now(datetime.timezone.utc)
    status = collect(github(os.environ.get("GITHUB_TOKEN", "")), args.repo, load_workflows(args.workflows), now)
    old = None
    if args.history and args.history.is_file() and args.history.stat().st_size:
        old = json.loads(args.history.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    (args.out / "history.json").write_text(
        json.dumps(update_history(old, history_point(status), now), indent=1) + "\n"
    )
    for w in status["critical"]:
        print(f"critical {w['file']}: " + (w.get("error") or f"{w['runs']} runs"))
    print("merge time: " + (status["merge_time"].get("error") or f"{status['merge_time']['merged']} merged"))
    for w in status["workflows"]:
        print(f"scheduled {w['file']}: " + (w.get("error") or f"{len(w['runs'])} runs"))
    errors = [w for w in status["critical"] + status["workflows"] if w.get("error")]
    # Everything failing means the token or the API is broken: say so with
    # the job's color. Some failing still publishes the rest.
    return 1 if len(errors) == len(status["critical"]) + len(status["workflows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
