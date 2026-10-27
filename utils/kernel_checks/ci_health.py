#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Record CI health for the maintainer dashboard.

Asks GitHub's API, with the workflow's own token, for three things and
writes them to one file, ``status.json``, which publishCiHealth.yml installs
as ``gh-pages:ci-health/status.json``; the dashboard reads it from there, so
no browser asks GitHub itself.

* **The critical test workflows** (``critical`` in ``ci_workflows.json``:
  those that gate a merge). Per workflow: its latest finished run on main
  (pushes and the nightly alike) and, when it has ``legs``, that run's
  jobs; and over the last ``WINDOW_DAYS`` days how often it passed on main,
  on pull requests and in the merge queue, how many commits needed a re-run
  to pass (flaky), how long a run on main took and waited for a runner; and
  every finished run on main in the window, for a chart of how long each
  took and waited.
  A run on main is judged by its jobs (``outcome``): a job that ran past
  its timeout fails it, though GitHub calls the run cancelled; a runner
  that was lost or never came is infrastructure, not a test failure; and a
  run cancelled by hand or by a newer one counts neither way. Its wait is
  its slowest job's, from the job's creation to its start, and its time
  from its first job's start to its end.
* **Merge time**: the pull requests merged in the window, how long each was
  open, and how many are open now and for how long.
* **The other scheduled jobs** (``workflows``): each one's recent scheduled
  runs on main and, for one marked ``legs``, the jobs of the latest, so a leg
  that failed or never found a runner can be named. A workflow is in one
  list or the other, never both.

* **Self-hosted runners**: each one named in the critical workflows' jobs
  in the window, and how many it ran that passed or failed. No separate
  request: GitHub's API does not say which runners are online without a
  repository-administrator token, which the publisher does not have.

It also keeps ``history.json``: one point a day of the critical workflows'
and merge time's figures, so the page can chart them. Pass the published
one with ``--history``; today's point is replaced as the day goes on. Pass
the published ``status.json`` with ``--previous``, so a run on main judged
before is not asked about again.

    GITHUB_TOKEN=... python3 utils/kernel_checks/ci_health.py --out ci-health \
        [--history published/history.json] [--previous published/status.json]

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
# Pages of 100 pull requests read, newest updated first.
PULL_PAGES = 5
# A conclusion that counts against a pass rate; cancelled and skipped do not.
FAILED = ("failure", "timed_out", "startup_failure")
# The critical workflows use these custom labels without "self-hosted".
SELF_HOSTED_LABELS = {"self-hosted", "aie2-4col", "aie2p-8col"}
RUNNERS_VERSION = 1
# A run on main's outcome, from its jobs. The first two count against its
# pass rate, the next two are the runners', and a cancel counts neither way.
OUTCOMES = ("failed", "timed_out", "runner_lost", "no_runner", "cancelled")
# What a job's annotations say when the job was stopped rather than failed,
# first match wins.
STOPPED = (
    ("exceeded the maximum execution time", "timed_out"),
    ("lost communication with the server", "runner_lost"),
    ("was canceled by", "cancelled"),
    ("higher priority waiting request", "cancelled"),
)

RUN_FIELDS = {
    "id": "id",
    "url": "html_url",
    "status": "status",
    "conclusion": "conclusion",
    "created_at": "created_at",
    "run_started_at": "run_started_at",
    "updated_at": "updated_at",
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
    "runner": "runner_name",
}
# What publishing keeps of a run on main beyond RUN_FIELDS, judged from its
# jobs once it finished.
JUDGED = ("outcome", "waited", "took", "runners", "runners_version")


def load_workflows(path: Path = WORKFLOWS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(item: dict, fields: dict) -> dict:
    return {ours: item.get(theirs) for ours, theirs in fields.items()}


class ApiError(Exception):
    pass


def github(token: str, timeout: float = 30, api: str = API):
    """Return a function that GETs one API path as JSON, with the token if any."""

    def get(path: str):
        req = urllib.request.Request(f"{api}{path}")
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
    """Parse an API timestamp as an aware datetime, or None."""
    if not stamp:
        return None
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def minutes(a, b):
    a, b = when(a), when(b)
    return None if a is None or b is None else (b - a).total_seconds() / 60


def percentile(values, q):
    """Return the nearest-rank ``q`` percentile of ``values``, or None."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    rank = max(1, min(len(xs), math.ceil(q * len(xs))))
    return xs[rank - 1]


def rounded(x, digits=1):
    return None if x is None else round(x, digits)


def pass_rate(runs):
    """Count completed runs as {passed, failed, rate}, rate the passed share."""
    passed = sum(r["conclusion"] == "success" for r in runs)
    failed = sum(r["conclusion"] in FAILED for r in runs)
    return {
        "passed": passed,
        "failed": failed,
        "rate": rounded(passed / (passed + failed), 3) if passed + failed else None,
    }


def main_rate(judged):
    """Count judged runs on main by outcome; rate passed over passed and failed.

    Returns:
        ``{passed, failed, timed_out, runner_lost, no_runner, cancelled,
        rate}``: a run past its timeout fails, a runner's loss and a cancel
        count neither way.
    """
    counts = {"passed": sum(r["outcome"] == "passed" for r in judged)}
    counts.update({o: sum(r["outcome"] == o for r in judged) for o in OUTCOMES})
    tried = counts["passed"] + counts["failed"] + counts["timed_out"]
    return {**counts, "rate": rounded(counts["passed"] / tried, 3) if tried else None}


def job_outcome(get, repo: str, job: dict) -> str:
    """Return what became of a finished job: passed, or one of OUTCOMES.

    A job that did not pass is told apart by its check run's annotations
    (one more request: a timeout, a lost runner, a cancel by hand or by a
    newer run), then by having no steps and no runner (it never found
    one), and otherwise by its conclusion.
    """
    if job.get("conclusion") in ("success", "skipped", "neutral"):
        return "passed"
    notes = " ".join(
        a.get("message") or ""
        for a in get(f"/repos/{repo}/check-runs/{job['id']}/annotations")
    )
    for text, outcome in STOPPED:
        if text in notes:
            return outcome
    if not job.get("steps") and not job.get("runner_name"):
        return "no_runner"
    if job.get("conclusion") == "cancelled":
        return "cancelled"
    return "timed_out" if job.get("conclusion") == "timed_out" else "failed"


def judge(get, repo: str, run: dict) -> dict:
    """Return a finished run on main's ``JUDGED`` fields, from its jobs.

    Its outcome is passed, or the first of OUTCOMES any of its jobs had (a
    run that did not pass with no job to say why is cancelled if GitHub
    says so and failed otherwise); its wait the longest any job waited for
    a runner, and its time from its first job's start to its end, in
    minutes. ``runners`` records passed, failed and timed-out jobs on
    self-hosted runners, excluding cancellations and infrastructure loss.
    """
    jobs = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100").get(
        "jobs", []
    )
    outcomes = [job_outcome(get, repo, j) for j in jobs]
    if run.get("conclusion") == "success":
        outcome = "passed"
    else:
        unexplained = "cancelled" if run.get("conclusion") == "cancelled" else "failed"
        outcome = next((o for o in OUTCOMES if o in outcomes), unexplained)
        if outcome == "cancelled" and run.get("conclusion") in FAILED:
            outcome = "failed"
    started = [j["started_at"] for j in jobs if j.get("started_at")]
    return {
        "outcome": outcome,
        "waited": rounded(
            max(
                (minutes(j.get("created_at"), j.get("started_at")) or 0 for j in jobs),
                default=None,
            )
        ),
        "took": (
            rounded(minutes(min(started), run.get("updated_at"))) if started else None
        ),
        "runners": [
            {"runner": j["runner_name"], "passed": result == "passed"}
            for j, result in zip(jobs, outcomes)
            if j.get("runner_name")
            and SELF_HOSTED_LABELS.intersection(j.get("labels") or [])
            and j.get("conclusion") not in ("skipped", "neutral")
            and result in ("passed", "failed", "timed_out")
        ],
        "runners_version": RUNNERS_VERSION,
    }


def jobs_of(get, repo: str, run_id) -> list:
    """Return a run's jobs, each finished one with its ``outcome``."""
    got = get(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")
    return [
        {
            **pick(j, JOB_FIELDS),
            **(
                {"outcome": job_outcome(get, repo, j)}
                if j.get("status") == "completed"
                else {}
            ),
        }
        for j in got.get("jobs", [])
    ]


# ----------------------------------------------------------------------
# The scheduled workflows
# ----------------------------------------------------------------------


def scheduled(get, repo: str, workflows: list) -> list:
    """Collect each workflow's recent scheduled runs on main.

    A workflow with legs also gets its latest run's jobs.
    """
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
                entry["jobs"] = jobs_of(get, repo, entry["runs"][0]["id"])
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
    """Count commits whose checks failed and then passed.

    That is a run that passed on a later attempt, or a commit with both a
    failed and a passed run. Returns {commits, flaky, rate} over the
    pull-request and merge-queue commits that finished.
    """
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
        retried = any(
            r["conclusion"] == "success" and (r.get("run_attempt") or 1) > 1 for r in rs
        )
        if retried or ("success" in conclusions and conclusions & set(FAILED)):
            shaky += 1
    n = len(by_sha)
    return {"commits": n, "flaky": shaky, "rate": rounded(shaky / n, 3) if n else None}


def critical(
    get, repo: str, workflows: list, now: datetime.datetime, judged: dict
) -> list:
    """Per critical workflow, its figures over the window.

    ``judged`` maps a run id to its ``JUDGED`` fields from an earlier
    publish, with its ``updated_at``; a run that has not changed since is
    not asked about again.
    """
    since = now - datetime.timedelta(days=WINDOW_DAYS)
    out = []
    for wf in workflows:
        entry = dict(wf)
        try:
            runs = window_runs(get, repo, wf["file"], since)
            done = [r for r in runs if r.get("status") == "completed"]
            on_main = [
                r
                for r in done
                if r.get("head_branch") == "main"
                and r.get("event") in ("push", "schedule")
            ]
            prs = [r for r in done if r.get("event") == "pull_request"]
            queue = [r for r in done if r.get("event") == "merge_group"]
            main_runs = []
            for r in on_main:
                seen = judged.get(r["id"])
                if (
                    not seen
                    or seen.get("updated_at") != r.get("updated_at")
                    or seen.get("runners_version") != RUNNERS_VERSION
                ):
                    seen = judge(get, repo, r)
                main_runs.append(
                    {**pick(r, RUN_FIELDS), **{k: seen[k] for k in JUDGED}}
                )
            took = [r["took"] for r in main_runs if r["outcome"] == "passed"]
            latest_main = on_main[0] if on_main else None
            # Its latest run on main may still be going: judged by the newest
            # one that is, which the page compares with `every` for lateness.
            newest_main = next(
                (
                    r
                    for r in runs
                    if r.get("head_branch") == "main"
                    and r.get("event") in ("push", "schedule")
                ),
                None,
            )
            main_jobs = None
            if wf.get("legs") and latest_main:
                main_jobs = jobs_of(get, repo, latest_main["id"])
            entry.update(
                runs=len(runs),
                main=main_rate(main_runs),
                latest_main=main_runs[0] if main_runs else None,
                newest_main=pick(newest_main, RUN_FIELDS) if newest_main else None,
                latest_main_jobs=main_jobs,
                main_runs=main_runs,
                pull_requests=pass_rate(prs),
                merge_queue=pass_rate(queue),
                flaky=flaky(runs),
                minutes_median=rounded(percentile(took, 0.5)),
                minutes_p90=rounded(percentile(took, 0.9)),
                queued_median=rounded(
                    percentile([r["waited"] for r in main_runs], 0.5)
                ),
            )
        except ApiError as e:
            entry["error"] = str(e)
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# Merge time
# ----------------------------------------------------------------------


def merge_time(get, repo: str, now: datetime.datetime) -> dict:
    """Measure the pull requests merged in the window and those open now.

    Merged ones by how long each was open; open ones by count and age.
    """
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
                    merged.append(
                        (m - (when(p["created_at"]) or m)).total_seconds() / 3600
                    )
            if len(got) < 100 or (when(got[-1]["updated_at"]) or since) < since:
                break
        open_ages, drafts = [], 0
        for page in range(1, PULL_PAGES + 1):
            got = get(f"/repos/{repo}/pulls?state=open&per_page=100&page={page}")
            for p in got:
                open_ages.append(
                    (now - (when(p["created_at"]) or now)).total_seconds() / 86400
                )
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
# Self-hosted runners
# ----------------------------------------------------------------------


def runner_tally(critical: list) -> dict:
    """Count each self-hosted runner's passed and failed jobs.

    From the critical workflows' runs on main in the window, already
    fetched to judge them: no extra requests. Returns
    ``{runner_name: {"passed", "failed"}}``.
    """
    tally = {}
    for wf in critical:
        for main_run in wf.get("main_runs") or []:
            for j in main_run.get("runners") or []:
                counts = tally.setdefault(j["runner"], {"passed": 0, "failed": 0})
                counts["passed" if j["passed"] else "failed"] += 1
    return tally


def runners(critical: list) -> list:
    """Each self-hosted runner named in the critical workflows' jobs.

    No request: ``runner_tally``'s counts, one entry per runner, by name.
    """
    return [
        {"name": name, **counts}
        for name, counts in sorted(runner_tally(critical).items())
    ]


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
    """Return ``old`` with today's point replaced or added.

    Points older than ``HISTORY_DAYS`` are dropped.
    """
    points = [
        p for p in ((old or {}).get("points") or []) if p.get("date") != point["date"]
    ]
    points.append(point)
    oldest = (now - datetime.timedelta(days=HISTORY_DAYS)).date().isoformat()
    points = sorted((p for p in points if p["date"] >= oldest), key=lambda p: p["date"])
    return {"schema": SCHEMA, "window_days": WINDOW_DAYS, "points": points}


def collect(
    get, repo: str, config: dict, now: datetime.datetime, previous=None
) -> dict:
    """Build the status file's contents.

    ``previous`` is the last published status, whose judged runs on main
    are kept rather than asked about again.
    """
    judged = {
        r["id"]: r
        for w in (previous or {}).get("critical", [])
        for r in w.get("main_runs") or []
        if all(k in r for k in JUDGED)
    }
    critical_wfs = critical(get, repo, config.get("critical", []), now, judged)
    return {
        "schema": SCHEMA,
        "generated_at": now.astimezone(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "repo": repo,
        "window_days": WINDOW_DAYS,
        "critical": critical_wfs,
        "merge_time": merge_time(get, repo, now),
        "workflows": scheduled(get, repo, config.get("workflows", [])),
        "runners": runners(critical_wfs),
    }


def main(argv=None, now: datetime.datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="directory for status.json and history.json",
    )
    parser.add_argument(
        "--history", type=Path, help="the published history.json, if any"
    )
    parser.add_argument(
        "--previous", type=Path, help="the published status.json, if any"
    )
    parser.add_argument("--api", default=API, help="the GitHub API to ask")
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", "Xilinx/mlir-aie")
    )
    parser.add_argument("--workflows", type=Path, default=WORKFLOWS)
    args = parser.parse_args(argv)

    now = now or datetime.datetime.now(datetime.timezone.utc)
    previous = None
    if args.previous and args.previous.is_file() and args.previous.stat().st_size:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
    status = collect(
        github(os.environ.get("GITHUB_TOKEN", ""), api=args.api),
        args.repo,
        load_workflows(args.workflows),
        now,
        previous,
    )
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
    print(
        "merge time: "
        + (
            status["merge_time"].get("error")
            or f"{status['merge_time']['merged']} merged"
        )
    )
    for w in status["workflows"]:
        print(f"scheduled {w['file']}: " + (w.get("error") or f"{len(w['runs'])} runs"))
    print(f"runners: {len(status['runners'])} named in the window")
    errors = [w for w in status["critical"] + status["workflows"] if w.get("error")]
    # Everything failing means the token or the API is broken: say so with
    # the job's color. Some failing still publishes the rest.
    return 1 if len(errors) == len(status["critical"]) + len(status["workflows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
