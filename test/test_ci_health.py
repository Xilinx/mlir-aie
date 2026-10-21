# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The CI health collector on canned API answers, served locally over HTTP."""

import datetime
import http.server
import importlib.util
import json
import re
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/kernel_checks/ci_health.py"
WORKFLOWS = ROOT / ".github/workflows"
NOW = datetime.datetime(2026, 10, 6, 12, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(scope="module")
def ci():
    spec = importlib.util.spec_from_file_location("ci_health", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def at(hours_ago):
    return (
        (NOW - datetime.timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")
    )


def run(
    id,
    hours_ago,
    event,
    conclusion,
    sha="a",
    attempt=1,
    took=30,
    waited=2,
    branch="main",
):
    start = hours_ago - waited / 60
    return {
        "id": id,
        "html_url": f"https://example.com/runs/{id}",
        "event": event,
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion,
        "created_at": at(hours_ago),
        "run_started_at": at(start),
        "updated_at": at(start - took / 60),
        "head_sha": sha,
        "head_branch": branch,
        "display_title": "t",
        "run_attempt": attempt,
        "repository": {"lots": "of fields"},
    }


def job(
    name,
    took,
    waited=1,
    conclusion="success",
    id=1,
    steps=3,
    runner="r1",
    hours_ago=5,
):
    return {
        "id": id,
        "name": name,
        "html_url": "u",
        "status": "completed",
        "conclusion": conclusion,
        "created_at": at(hours_ago),
        "started_at": at(hours_ago - waited / 60),
        "completed_at": at(hours_ago - (waited + took) / 60),
        "steps": [{"name": f"step {i}"} for i in range(steps)],
        "runner_name": runner,
    }


class Served:
    """A local GitHub API: answers by path pattern, and records what was asked.

    An answer that is an int is that HTTP status, with no body.
    """

    def __init__(self, routes):
        self.routes, self.asked = routes, []
        served = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                served.asked.append(self.path)
                answer = next(
                    (a for p, a in served.routes if re.search(p, self.path)), 404
                )
                body = b"" if isinstance(answer, int) else json.dumps(answer).encode()
                self.send_response(answer if isinstance(answer, int) else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def served(ci):
    servers = []

    def serve(routes):
        servers.append(Served(routes))
        return ci.github("", api=servers[-1].url), servers[-1]

    yield serve
    for s in servers:
        s.server.shutdown()


def test_the_workflow_lists_name_real_workflows(ci):
    config = ci.load_workflows()
    for wf in config["workflows"]:
        text = (WORKFLOWS / wf["file"]).read_text(encoding="utf-8")
        assert re.search(r"^\s+schedule:", text, re.M), f"{wf['file']} has no schedule"
        assert wf["every"] > 0 and wf["title"] and wf["why"]
    # The critical ones gate a merge: they run in the merge queue.
    assert config["critical"]
    for wf in config["critical"]:
        text = (WORKFLOWS / wf["file"]).read_text(encoding="utf-8")
        assert re.search(
            r"^  merge_group:", text, re.M
        ), f"{wf['file']} is not in the merge queue"
        assert wf["every"] > 0 and wf["title"]
    # Each workflow is shown once: in one list, never both.
    files = [w["file"] for w in config["critical"] + config["workflows"]]
    assert len(files) == len(set(files))


def test_percentiles_and_pass_rates(ci):
    assert ci.percentile([], 0.5) is None
    assert ci.percentile([5, 1, 3], 0.5) == 3
    assert ci.percentile(range(1, 11), 0.9) == 9
    assert ci.percentile([1, None, 2], 0.5) == 1
    runs = [
        {"conclusion": c}
        for c in ("success", "success", "failure", "cancelled", "timed_out")
    ]
    # Cancelled runs count neither way.
    assert ci.pass_rate(runs) == {"passed": 2, "failed": 2, "rate": 0.5}
    assert ci.pass_rate([]) == {"passed": 0, "failed": 0, "rate": None}


def test_a_commit_that_needed_a_rerun_is_flaky(ci):
    runs = [
        run(1, 5, "pull_request", "success", sha="steady"),
        run(2, 5, "pull_request", "success", sha="retried", attempt=2),
        run(3, 6, "pull_request", "failure", sha="twice"),
        run(4, 5, "merge_group", "success", sha="twice"),
        run(5, 5, "pull_request", "failure", sha="broken"),
        # Pushes to main are not commits under review.
        run(6, 5, "push", "success", sha="main", attempt=3),
    ]
    assert ci.flaky(runs) == {"commits": 4, "flaky": 2, "rate": 0.5}


def critical_routes():
    window = [
        run(10, 2, "push", "failure", sha="m2"),
        run(9, 30, "push", "success", sha="m1", took=60),
        run(8, 4, "pull_request", "success", sha="p1", attempt=2, took=40),
        run(7, 6, "merge_group", "success", sha="q1", took=50),
        run(6, 7, "pull_request", "failure", sha="p2"),
        # Before the window: left out.
        run(5, 24 * 20, "push", "failure", sha="old"),
    ]
    return [
        (
            r"/actions/workflows/crit\.yml/runs\?created=%3E%3D2026-09-22",
            {"workflow_runs": window},
        ),
        (r"/actions/runs/10/jobs", {"jobs": [job("build", 30, conclusion="failure")]}),
        (
            r"/actions/runs/9/jobs",
            {
                "jobs": [
                    job("build", 50, hours_ago=30),
                    job("test", 9, 0.5, hours_ago=30),
                ]
            },
        ),
        (r"/actions/runs/\d+/jobs", {"jobs": [job("build", 30), job("test", 10)]}),
        (
            r"/check-runs/\d+/annotations",
            [{"message": "Process completed with exit code 1."}],
        ),
    ]


def test_a_critical_workflow_is_measured_over_the_window(ci, served):
    get, api = served(critical_routes())
    (w,) = ci.critical(
        get, "o/r", [{"file": "crit.yml", "title": "Crit", "legs": True}], NOW, {}
    )
    assert w["runs"] == 5
    assert w["main"] == {
        "passed": 1,
        "failed": 1,
        "timed_out": 0,
        "runner_lost": 0,
        "no_runner": 0,
        "cancelled": 0,
        "rate": 0.5,
    }
    assert w["latest_main"]["id"] == 10 and w["latest_main"]["conclusion"] == "failure"
    assert set(w["latest_main"]) == set(ci.RUN_FIELDS) | set(ci.JUDGED)
    assert w["latest_main"]["outcome"] == "failed"
    assert w["pull_requests"] == {"passed": 1, "failed": 1, "rate": 0.5}
    assert w["merge_queue"] == {"passed": 1, "failed": 0, "rate": 1.0}
    assert w["flaky"] == {"commits": 3, "flaky": 1, "rate": 0.333}
    # Runs on main only, from the first job's start (half a minute after the
    # run was created) to its end: the pull-request and merge-queue runs and
    # the wait for a runner are not in it.
    passed = w["main_runs"][1]
    assert passed["took"] == pytest.approx(60 + 2 - 0.5, abs=0.1)
    assert (w["minutes_median"], w["minutes_p90"]) == (passed["took"],) * 2
    # The wait is the slowest job's, from its creation to its start.
    assert w["queued_median"] == 1.0
    # With legs, the latest run on main's jobs, each with what became of it.
    assert [(j["name"], j["outcome"]) for j in w["latest_main_jobs"]] == [
        ("build", "failed")
    ]
    assert w["latest_main_jobs"][0]["runner"] == "r1"
    assert w["newest_main"]["id"] == 10
    # Its finished runs on main, newest first, each with when it ended.
    assert [r["id"] for r in w["main_runs"]] == [10, 9]
    assert w["main_runs"][1]["updated_at"] == at(30 - 2 / 60 - 1)


TIMED_OUT = (
    "The job running on runner aie9 has exceeded the maximum execution time of 30m0s."
)
RUNNER_LOST = (
    "The self-hosted runner: aie-win lost communication with the server. Verify the"
    " machine is running and has a healthy network connection."
)


@pytest.mark.parametrize(
    "conclusion, jobs, notes, outcome",
    [
        # GitHub calls a run whose job ran past its timeout cancelled.
        (
            "cancelled",
            [job("test", 30, conclusion="cancelled")],
            TIMED_OUT,
            "timed_out",
        ),
        ("failure", [job("test", 3, conclusion="failure")], RUNNER_LOST, "runner_lost"),
        # Queued until it was cancelled: no runner, no step.
        (
            "cancelled",
            [job("test", 0, conclusion="cancelled", steps=0, runner="")],
            "The operation was canceled.",
            "no_runner",
        ),
        (
            "cancelled",
            [job("test", 3, conclusion="cancelled")],
            "The run was canceled by @someone.",
            "cancelled",
        ),
        (
            "failure",
            [job("test", 3, conclusion="failure"), job("lint", 1, id=2)],
            "Process completed with exit code 1.",
            "failed",
        ),
    ],
)
def test_a_run_on_main_is_judged_by_its_jobs(
    ci, served, conclusion, jobs, notes, outcome
):
    window = [run(10, 2, "schedule", conclusion), run(9, 30, "schedule", "success")]
    get, _ = served(
        [
            (r"/workflows/crit\.yml/runs", {"workflow_runs": window}),
            (r"/actions/runs/10/jobs", {"jobs": jobs}),
            (r"/actions/runs/9/jobs", {"jobs": [job("test", 30)]}),
            (r"/check-runs/1/annotations", [{"message": notes}]),
            (r"/check-runs/\d+/annotations", []),
        ]
    )
    (w,) = ci.critical(get, "o/r", [{"file": "crit.yml", "title": "Crit"}], NOW, {})
    assert w["main_runs"][0]["outcome"] == outcome
    # Only a failed test or a timeout counts against the rate.
    counts = w["main"]
    assert counts[outcome] == 1 and counts["passed"] == 1
    assert counts["rate"] == (0.5 if outcome in ("failed", "timed_out") else 1.0)


def test_a_run_judged_before_is_not_asked_about_again(ci, served):
    get, api = served(critical_routes())
    config = [{"file": "crit.yml", "title": "Crit"}]
    (first,) = ci.critical(get, "o/r", config, NOW, {})
    asked = len(api.asked)
    judged = {r["id"]: r for r in first["main_runs"]}
    (again,) = ci.critical(get, "o/r", config, NOW, judged)
    assert again["main_runs"] == first["main_runs"]
    # Only the window's runs list again; no run's jobs.
    assert [p for p in api.asked[asked:] if "/jobs" in p] == []


def test_runs_are_read_page_by_page_until_a_short_one(ci, served):
    full = {"workflow_runs": [run(i, 1, "push", "success") for i in range(100)]}
    short = {"workflow_runs": [run(500, 1, "push", "success")]}
    get, api = served([(r"page=1$", full), (r"page=2$", short)])
    assert (
        len(ci.window_runs(get, "o/r", "w.yml", NOW - datetime.timedelta(days=14)))
        == 101
    )
    assert len(api.asked) == 2


def test_merge_time_counts_the_window_and_the_open_ones(ci, served):
    def pr(created_h, merged_h=None, updated_h=None, draft=False):
        return {
            "created_at": at(created_h),
            "merged_at": at(merged_h) if merged_h is not None else None,
            "updated_at": at(updated_h if updated_h is not None else (merged_h or 0)),
            "draft": draft,
        }

    closed = [pr(30, 6), pr(100, 4), pr(10, None, updated_h=3), pr(24 * 40, 24 * 20)]
    opened = [pr(24), pr(24 * 45, draft=True)]
    get, _ = served([(r"state=closed", closed), (r"state=open", opened)])
    m = ci.merge_time(get, "o/r", NOW)
    # 24 and 96 hours; the one merged 20 days ago is out of the window.
    assert m == {
        "merged": 2,
        "hours_median": 24.0,
        "hours_p90": 96.0,
        "open": 2,
        "open_drafts": 1,
        "open_over_30_days": 1,
    }
    get, _ = served([(r"pulls", 403)])
    error = ci.merge_time(get, "o/r", NOW)["error"]
    assert error.startswith("HTTP 403 for /repos/o/r/pulls?state=closed")


def test_the_status_file_and_one_failure_does_not_stop_the_rest(ci, served):
    routes = [
        (r"/workflows/bad\.yml/", 404),
        (
            r"/workflows/nightly\.yml/runs\?branch=main&event=schedule",
            {"workflow_runs": [run(20, 3, "schedule", "success")]},
        ),
        (r"/pulls\?state=", []),
    ] + critical_routes()
    config = {
        "critical": [
            {"file": "crit.yml", "title": "Crit"},
            {"file": "bad.yml", "title": "Bad"},
        ],
        "workflows": [
            {
                "file": "nightly.yml",
                "title": "N",
                "group": "G",
                "every": 24,
                "legs": True,
                "why": "w",
            }
        ],
    }
    get, _ = served(routes)
    status = ci.collect(get, "o/r", config, NOW)
    assert (
        status["schema"] == 1
        and status["generated_at"] == "2026-10-06T12:00:00+00:00"
        and status["window_days"] == 14
    )
    bad = status["critical"][1]
    assert set(bad) == {"file", "title", "error"}
    assert bad["error"].startswith("HTTP 404 for /repos/o/r/actions/workflows/bad.yml/")
    assert status["critical"][0]["main"]["rate"] == 0.5
    (nightly,) = status["workflows"]
    assert [r["id"] for r in nightly["runs"]] == [20]
    assert [j["name"] for j in nightly["jobs"]] == ["build", "test"]
    assert status["merge_time"]["merged"] == 0
    # The point a day the page charts.
    point = ci.history_point(status)
    assert point["date"] == "2026-10-06"
    assert point["critical"] == {
        "crit.yml": {
            "main": 0.5,
            "pull_requests": 0.5,
            "flaky": 0.333,
            "minutes_median": status["critical"][0]["minutes_median"],
            "queued_median": 1.0,
        }
    }


def test_history_keeps_a_point_a_day_for_half_a_year(ci):
    old = {
        "schema": 1,
        "points": [
            {"date": "2026-01-01"},
            {"date": "2026-10-05", "v": 1},
            {"date": "2026-10-06", "v": 1},
        ],
    }
    new = ci.update_history(old, {"date": "2026-10-06", "v": 2}, NOW)
    assert new["points"] == [
        {"date": "2026-10-05", "v": 1},
        {"date": "2026-10-06", "v": 2},
    ]
    assert ci.update_history(None, {"date": "2026-10-06"}, NOW)["points"] == [
        {"date": "2026-10-06"}
    ]


def test_main_writes_both_files_and_fails_only_when_nothing_answered(
    ci, tmp_path, served
):
    config = tmp_path / "w.json"
    config.write_text(
        json.dumps(
            {"critical": [{"file": "crit.yml", "title": "Crit"}], "workflows": []}
        )
    )
    (tmp_path / "old.json").write_text(
        json.dumps({"schema": 1, "points": [{"date": "2026-10-05"}]})
    )
    _, api = served(critical_routes() + [(r"/pulls", [])])
    out = tmp_path / "out"
    assert (
        ci.main(
            [
                "--out",
                str(out),
                "--workflows",
                str(config),
                "--history",
                str(tmp_path / "old.json"),
                "--api",
                api.url,
            ],
            now=NOW,
        )
        == 0
    )
    status = json.loads((out / "status.json").read_text())
    assert status["critical"][0]["file"] == "crit.yml"
    dates = [
        p["date"] for p in json.loads((out / "history.json").read_text())["points"]
    ]
    assert dates[0] == "2026-10-05" and len(dates) == 2
    # The published status is read back: its runs on main are not judged again.
    _, again = served(critical_routes() + [(r"/pulls", [])])
    previous = ["--previous", str(out / "status.json"), "--api", again.url]
    assert (
        ci.main(["--out", str(out), "--workflows", str(config), *previous], now=NOW)
        == 0
    )
    assert not any("/jobs" in p for p in again.asked)
    _, refusing = served([(r".", 401)])
    assert (
        ci.main(
            ["--out", str(out), "--workflows", str(config), "--api", refusing.url],
            now=NOW,
        )
        == 1
    )
