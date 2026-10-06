# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The CI health collector on canned API answers; no network."""

import datetime
import importlib.util
import json
import re
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
    return (NOW - datetime.timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def run(id, hours_ago, event, conclusion, sha="a", attempt=1, took=30, waited=2, branch="main"):
    start = hours_ago - waited / 60
    return {
        "id": id, "html_url": f"https://example.com/runs/{id}", "event": event,
        "status": "completed" if conclusion else "in_progress", "conclusion": conclusion,
        "created_at": at(hours_ago), "run_started_at": at(start), "updated_at": at(start - took / 60),
        "head_sha": sha, "head_branch": branch, "display_title": "t", "run_attempt": attempt,
        "repository": {"lots": "of fields"},
    }


def job(name, took, waited=1, conclusion="success"):
    return {
        "name": name, "html_url": "u", "status": "completed", "conclusion": conclusion,
        "created_at": at(5), "started_at": at(5 - waited / 60), "completed_at": at(5 - (waited + took) / 60),
    }


class FakeGitHub:
    """Answers by path pattern; records what was asked."""

    def __init__(self, routes):
        self.routes, self.asked = routes, []

    def __call__(self, path):
        self.asked.append(path)
        for pattern, answer in self.routes:
            if re.search(pattern, path):
                if isinstance(answer, Exception):
                    raise answer
                return answer(path) if callable(answer) else answer
        raise AssertionError(f"unexpected request {path}")


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
        assert re.search(r"^  merge_group:", text, re.M), f"{wf['file']} is not in the merge queue"
        assert wf["every"] > 0 and wf["title"]
    # Each workflow is shown once: in one list, never both.
    files = [w["file"] for w in config["critical"] + config["workflows"]]
    assert len(files) == len(set(files))


def test_percentiles_and_pass_rates(ci):
    assert ci.percentile([], 0.5) is None
    assert ci.percentile([5, 1, 3], 0.5) == 3
    assert ci.percentile(range(1, 11), 0.9) == 9
    assert ci.percentile([1, None, 2], 0.5) == 1
    runs = [{"conclusion": c} for c in ("success", "success", "failure", "cancelled", "timed_out")]
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


def test_slowest_jobs_are_timed_from_the_sampled_runs(ci):
    sampled = [
        [job("build", 40), job("test", 20, waited=10), job("lint", 2)],
        [job("build", 50), job("test", 25, waited=30, conclusion="failure"), {**job("skip", 99), "conclusion": "skipped"}],
    ]
    rows = ci.slowest_jobs(sampled)
    assert [r["name"] for r in rows] == ["build", "test", "lint"]
    assert rows[0] == {"name": "build", "n": 2, "minutes_median": 40.0, "minutes_p90": 50.0, "queued_median": 1.0, "failed": 0}
    assert rows[1]["failed"] == 1 and rows[1]["queued_median"] == 10.0


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
        (r"/actions/workflows/crit\.yml/runs\?created=%3E%3D2026-09-22", {"workflow_runs": window}),
        (r"/actions/runs/\d+/jobs", {"jobs": [job("build", 30), job("test", 10)]}),
    ]


def test_a_critical_workflow_is_measured_over_the_window(ci):
    get = FakeGitHub(critical_routes())
    (w,) = ci.critical(get, "o/r", [{"file": "crit.yml", "title": "Crit", "legs": True}], NOW)
    assert w["runs"] == 5
    assert w["main"] == {"passed": 1, "failed": 1, "rate": 0.5}
    assert w["latest_main"]["id"] == 10 and w["latest_main"]["conclusion"] == "failure"
    assert set(w["latest_main"]) == set(ci.RUN_FIELDS)
    assert w["pull_requests"] == {"passed": 1, "failed": 1, "rate": 0.5}
    assert w["merge_queue"] == {"passed": 1, "failed": 0, "rate": 1.0}
    assert w["flaky"] == {"commits": 3, "flaky": 1, "rate": 0.333}
    # Passing runs only: 60, 40 and 50 minutes.
    assert (w["minutes_median"], w["minutes_p90"]) == (50.0, 60.0)
    assert w["queued_median"] == 2.0
    assert w["jobs_sampled"] == 5
    # With legs, the latest run on main's jobs, to name the one that failed.
    assert [j["name"] for j in w["latest_main_jobs"]] == ["build", "test"]
    assert w["newest_main"]["id"] == 10
    assert [j["name"] for j in w["slowest_jobs"]] == ["build", "test"]


def test_runs_are_read_page_by_page_until_a_short_one(ci):
    full = {"workflow_runs": [run(i, 1, "push", "success") for i in range(100)]}
    short = {"workflow_runs": [run(500, 1, "push", "success")]}
    get = FakeGitHub([(r"page=1$", full), (r"page=2$", short)])
    assert len(ci.window_runs(get, "o/r", "w.yml", NOW - datetime.timedelta(days=14))) == 101
    assert len(get.asked) == 2


def test_merge_time_counts_the_window_and_the_open_ones(ci):
    def pr(created_h, merged_h=None, updated_h=None, draft=False):
        return {"created_at": at(created_h), "merged_at": at(merged_h) if merged_h is not None else None,
                "updated_at": at(updated_h if updated_h is not None else (merged_h or 0)), "draft": draft}

    closed = [pr(30, 6), pr(100, 4), pr(10, None, updated_h=3), pr(24 * 40, 24 * 20)]
    opened = [pr(24), pr(24 * 45, draft=True)]
    get = FakeGitHub([(r"state=closed", closed), (r"state=open", opened)])
    m = ci.merge_time(get, "o/r", NOW)
    # 24 and 96 hours; the one merged 20 days ago is out of the window.
    assert m == {"merged": 2, "hours_median": 24.0, "hours_p90": 96.0, "open": 2, "open_drafts": 1, "open_over_30_days": 1}
    assert ci.merge_time(FakeGitHub([(r"pulls", ci.ApiError("HTTP 403 for /pulls"))]), "o/r", NOW) == {"error": "HTTP 403 for /pulls"}


def test_the_status_file_and_one_failure_does_not_stop_the_rest(ci):
    routes = critical_routes() + [
        (r"/workflows/bad\.yml/", ci.ApiError("HTTP 404 for bad")),
        (r"/workflows/nightly\.yml/runs\?branch=main&event=schedule", {"workflow_runs": [run(20, 3, "schedule", "success")]}),
        (r"/pulls\?state=", []),
    ]
    config = {
        "critical": [{"file": "crit.yml", "title": "Crit"}, {"file": "bad.yml", "title": "Bad"}],
        "workflows": [{"file": "nightly.yml", "title": "N", "group": "G", "every": 24, "legs": True, "why": "w"}],
    }
    status = ci.collect(FakeGitHub(routes), "o/r", config, NOW)
    assert status["schema"] == 1 and status["generated_at"] == "2026-10-06T12:00:00+00:00" and status["window_days"] == 14
    assert status["critical"][1] == {"file": "bad.yml", "title": "Bad", "error": "HTTP 404 for bad"}
    assert status["critical"][0]["main"]["rate"] == 0.5
    (nightly,) = status["workflows"]
    assert [r["id"] for r in nightly["runs"]] == [20]
    assert [j["name"] for j in nightly["jobs"]] == ["build", "test"]
    assert status["merge_time"]["merged"] == 0
    # The point a day the page charts.
    point = ci.history_point(status)
    assert point["date"] == "2026-10-06"
    assert point["critical"] == {"crit.yml": {"main": 0.5, "pull_requests": 0.5, "flaky": 0.333, "minutes_median": 50.0, "queued_median": 2.0}}


def test_history_keeps_a_point_a_day_for_half_a_year(ci):
    old = {"schema": 1, "points": [{"date": "2026-01-01"}, {"date": "2026-10-05", "v": 1}, {"date": "2026-10-06", "v": 1}]}
    new = ci.update_history(old, {"date": "2026-10-06", "v": 2}, NOW)
    assert new["points"] == [{"date": "2026-10-05", "v": 1}, {"date": "2026-10-06", "v": 2}]
    assert ci.update_history(None, {"date": "2026-10-06"}, NOW)["points"] == [{"date": "2026-10-06"}]


def test_main_writes_both_files_and_fails_only_when_nothing_answered(ci, tmp_path, monkeypatch):
    config = tmp_path / "w.json"
    config.write_text(json.dumps({"critical": [{"file": "crit.yml", "title": "Crit"}], "workflows": []}))
    (tmp_path / "old.json").write_text(json.dumps({"schema": 1, "points": [{"date": "2026-10-05"}]}))
    monkeypatch.setattr(ci, "github", lambda token: FakeGitHub(critical_routes() + [(r"/pulls", [])]))
    out = tmp_path / "out"
    assert ci.main(["--out", str(out), "--workflows", str(config), "--history", str(tmp_path / "old.json")]) == 0
    status = json.loads((out / "status.json").read_text())
    assert status["critical"][0]["file"] == "crit.yml"
    dates = [p["date"] for p in json.loads((out / "history.json").read_text())["points"]]
    assert dates[0] == "2026-10-05" and len(dates) == 2
    monkeypatch.setattr(ci, "github", lambda token: FakeGitHub([(r".", ci.ApiError("HTTP 401"))]))
    assert ci.main(["--out", str(out), "--workflows", str(config)]) == 1
