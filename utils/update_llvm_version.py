#!/usr/bin/env python3

import argparse
import re
import sys
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Constants
TRITON_LLVM_HASH_URL = (
    "https://raw.githubusercontent.com/triton-lang/triton/main/cmake/llvm-hash.txt"
)
TORCH_MLIR_SUBMODULE_URL = (
    "https://api.github.com/repos/llvm/torch-mlir/contents/externals/llvm-project"
)
LLVM_PROJECT_COMMIT_URL = "https://api.github.com/repos/llvm/llvm-project/commits/{}"
EUDSL_INDEX_URL = "https://llvm.github.io/eudsl/"
EUDSL_SUBMODULE_URL = (
    "https://api.github.com/repos/llvm/eudsl/contents/third_party/llvm-project?ref={}"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONE_LLVM_SH = REPO_ROOT / "utils" / "clone-llvm.sh"
REQUIREMENTS_TXT = REPO_ROOT / "python" / "requirements.txt"

COMMIT_DATE_CACHE = {}


def get_request(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req)


def populate_commit_date_cache(center_date, days=2):
    """
    Fetches commits around the center_date to populate the cache.
    This reduces individual API calls for commit dates.
    """
    since = (center_date - timedelta(days=days)).isoformat()
    until = (center_date + timedelta(days=days)).isoformat()

    url = "https://api.github.com/repos/llvm/llvm-project/commits"
    params = {
        "since": since,
        "until": until,
        "per_page": 100,
    }

    print(f"Prefetching LLVM commits between {since} and {until}...")
    try:
        with get_request(url, params) as response:
            commits = json.loads(response.read().decode("utf-8"))
            for commit in commits:
                sha = commit["sha"]
                date_str = commit["commit"]["committer"]["date"]
                dt = parse_utc_date(date_str)
                COMMIT_DATE_CACHE[sha] = dt
            print(f"Cached {len(commits)} commits.")
    except Exception as e:
        print(f"Error prefetching commits: {e}", file=sys.stderr)


def get_request_simple(url):
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req)


def check_token():
    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "Warning: GITHUB_TOKEN not set. API rate limits may be exceeded.",
            file=sys.stderr,
        )


def parse_utc_date(date_str):
    # Handles 2025-12-05T18:00:00Z
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def get_commit_date(commit_hash):
    if not commit_hash:
        return None
    if commit_hash in COMMIT_DATE_CACHE:
        return COMMIT_DATE_CACHE[commit_hash]

    url = LLVM_PROJECT_COMMIT_URL.format(commit_hash)
    try:
        with get_request_simple(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            date_str = data["commit"]["committer"]["date"]
            dt = parse_utc_date(date_str)
            COMMIT_DATE_CACHE[commit_hash] = dt
            return dt
    except Exception as e:
        print(f"Error fetching commit date for {commit_hash}: {e}", file=sys.stderr)
        return None


def get_current_llvm_commit():
    if not CLONE_LLVM_SH.exists():
        print(f"Error: {CLONE_LLVM_SH} does not exist.", file=sys.stderr)
        return None
    content = CLONE_LLVM_SH.read_text()
    match = re.search(r"LLVM_PROJECT_COMMIT=([a-f0-9]+)", content)
    if match:
        return match.group(1)
    return None


def get_triton_llvm_commit():
    try:
        with get_request_simple(TRITON_LLVM_HASH_URL) as response:
            content = response.read().decode("utf-8").strip()
            if re.match(r"^[a-f0-9]+$", content):
                return content
    except Exception as e:
        print(f"Error fetching Triton LLVM commit: {e}", file=sys.stderr)
    return None


def get_torch_mlir_llvm_commit():
    try:
        with get_request_simple(TORCH_MLIR_SUBMODULE_URL) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["sha"]
    except Exception as e:
        print(f"Error fetching Torch-MLIR LLVM commit: {e}", file=sys.stderr)
    return None


def get_eudsl_candidates(target_date, window_days=14):
    candidates = []
    try:
        with urllib.request.urlopen(EUDSL_INDEX_URL) as response:
            content = response.read().decode("utf-8")
            # Regex: eudsl_python_extras-0\.1\.0\.(\d+\.\d+)\+([a-f0-9]+)\.tar\.gz
            matches = re.findall(
                r"eudsl_python_extras-0\.1\.0\.(\d+\.\d+)\+([a-f0-9]+)\.tar\.gz",
                content,
            )

            for date_str, eudsl_hash in matches:
                # date_str is YYYYMMDD.HHMM
                try:
                    # Parse as UTC
                    # The date in version string is likely UTC or close to it.
                    version_date = datetime.strptime(date_str, "%Y%m%d.%H%M").replace(
                        tzinfo=timezone.utc
                    )

                    # Check if within window
                    diff = abs((version_date - target_date).days)
                    if diff <= window_days:
                        full_version = f"0.1.0.{date_str}+{eudsl_hash}"
                        candidates.append((version_date, eudsl_hash, full_version))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error fetching eudsl versions: {e}", file=sys.stderr)
        return []
    return candidates


def find_closest_eudsl_version(target_llvm_hash, target_llvm_date):
    print(
        f"Looking for eudsl version with LLVM date close to {target_llvm_hash[:8]} ({target_llvm_date})..."
    )

    # 1. Populate cache with commits around target date
    populate_commit_date_cache(target_llvm_date, days=3)

    # 2. Get candidates based on date
    # We look for eudsl versions released around the target date.
    # Reduced window to 7 days to reduce API calls
    candidates = get_eudsl_candidates(target_llvm_date, window_days=7)
    print(f"Found {len(candidates)} eudsl candidates within date range.")

    best_candidate = None
    min_diff = timedelta.max

    for ver_date, eudsl_hash, full_version in candidates:
        # Get LLVM hash for this eudsl version
        try:
            url = EUDSL_SUBMODULE_URL.format(eudsl_hash)
            with get_request_simple(url) as response:
                data = json.loads(response.read().decode("utf-8"))
                llvm_hash = data["sha"]

                # Get date of this LLVM hash
                llvm_date = get_commit_date(llvm_hash)
                if not llvm_date:
                    continue

                # Compare with target_llvm_date
                diff = abs(llvm_date - target_llvm_date)

                # print(f"  Checked {full_version}: LLVM {llvm_hash[:8]} ({llvm_date}), diff: {diff}")

                if diff < min_diff:
                    min_diff = diff
                    best_candidate = (llvm_hash, llvm_date, full_version)

        except Exception as e:
            print(f"Error checking eudsl candidate {eudsl_hash}: {e}", file=sys.stderr)

    return best_candidate


MLIR_DISTRO_RELEASE_URL = (
    "https://api.github.com/repos/Xilinx/mlir-aie/releases/tags/mlir-distro"
)


def find_wheel_in_distro(commit_hash):
    """Query the mlir-distro GitHub release for a wheel matching this LLVM commit.

    The mlirDistro.yml workflow stamps wheels with the CI runner's wall-clock
    time (``date +"%Y%m%d%H"``), and reads the LLVM major.minor.patch version
    from ``LLVMVersion.cmake``.  Both values are baked into the wheel filename.
    Since neither can be derived from the LLVM commit metadata alone, we look
    up the actual wheel to get the correct version string.

    Uses the paginated release assets endpoint to handle releases with many
    assets (the mlir-distro release accumulates wheels over time).

    Returns (major_minor_patch, datetime_str) or None.
    """
    commit_short = commit_hash[:8]
    # Match base 'mlir-' wheels (not mlir_no_rtti or mlir_native_tools).
    pattern = re.compile(
        rf"^mlir-(\d+\.\d+\.\d+)\.(\d{{10}})\+{re.escape(commit_short)}-"
    )
    try:
        # First, get the release ID.
        with get_request_simple(MLIR_DISTRO_RELEASE_URL) as response:
            release_data = json.loads(response.read().decode("utf-8"))
        release_id = release_data["id"]

        # Paginate through the release assets endpoint.
        page = 1
        while True:
            assets_url = (
                f"https://api.github.com/repos/Xilinx/mlir-aie"
                f"/releases/{release_id}/assets?per_page=100&page={page}"
            )
            with get_request_simple(assets_url) as response:
                assets = json.loads(response.read().decode("utf-8"))
            if not assets:
                break
            for asset in assets:
                m = pattern.match(asset["name"])
                if m:
                    version, datetime_str = m.group(1), m.group(2)
                    print(
                        f"Found wheel in mlir-distro: "
                        f"mlir-{version}.{datetime_str}+{commit_short}"
                    )
                    return version, datetime_str
            if len(assets) < 100:
                break
            page += 1

        print(
            f"Warning: No wheel found in mlir-distro for commit {commit_short}.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error querying mlir-distro release: {e}", file=sys.stderr)
    return None


def update_files(
    new_commit, commit_date, eudsl_version, wheel_version=None, wheel_datetime=None
):
    # Use wheel-derived values when available, fall back to commit date.
    if wheel_datetime:
        datetime_str = wheel_datetime
    else:
        # Fallback: derive from LLVM commit date (may not match the actual
        # wheel, since mlirDistro.yml uses CI build time instead).
        datetime_str = commit_date.strftime("%Y%m%d%H")
        print(
            "Warning: Using LLVM commit date as DATETIME fallback. "
            "This may not match the actual wheel version.",
            file=sys.stderr,
        )

    print(f"Updating to LLVM commit: {new_commit}")
    print(f"Commit Date: {commit_date}")
    print(f"DATETIME: {datetime_str}")
    if wheel_version:
        print(f"WHEEL_VERSION prefix: {wheel_version}")
    print(f"EUDSL Version: {eudsl_version}")

    # Update utils/clone-llvm.sh
    if CLONE_LLVM_SH.exists():
        content = CLONE_LLVM_SH.read_text()
        content = re.sub(
            r"LLVM_PROJECT_COMMIT=[a-f0-9]+",
            f"LLVM_PROJECT_COMMIT={new_commit}",
            content,
        )
        content = re.sub(r"DATETIME=\d+", f"DATETIME={datetime_str}", content)
        # Update the major.minor.patch prefix in WHEEL_VERSION (e.g. 22.0.0 -> 23.0.0).
        if wheel_version:
            content = re.sub(
                r"(WHEEL_VERSION=)\d+\.\d+\.\d+",
                rf"\g<1>{wheel_version}",
                content,
            )
        CLONE_LLVM_SH.write_text(content)
        print(f"Updated {CLONE_LLVM_SH}")
    else:
        print(f"Error: {CLONE_LLVM_SH} not found.", file=sys.stderr)

    # Update python/requirements.txt
    if REQUIREMENTS_TXT.exists():
        content = REQUIREMENTS_TXT.read_text()
        # Regex to match eudsl-python-extras==...
        pattern = r"(eudsl-python-extras==)[0-9\.\+a-z]+"
        if re.search(pattern, content):
            content = re.sub(pattern, f"\\g<1>{eudsl_version}", content)
            REQUIREMENTS_TXT.write_text(content)
            print(f"Updated {REQUIREMENTS_TXT}")
        else:
            print(
                f"Warning: Could not find eudsl-python-extras in {REQUIREMENTS_TXT}",
                file=sys.stderr,
            )
    else:
        print(f"Error: {REQUIREMENTS_TXT} not found.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Update LLVM version based on Triton or Torch-MLIR."
    )
    parser.add_argument(
        "--llvm-hash", help="Specific LLVM hash to use (overrides auto-detection)."
    )
    args = parser.parse_args()

    check_token()

    # 1. Get current info
    current_commit = get_current_llvm_commit()
    if not current_commit:
        print("Could not determine current LLVM commit. Aborting.", file=sys.stderr)
        sys.exit(1)

    current_date = get_commit_date(current_commit)
    if not current_date:
        print("Could not determine date of current LLVM commit. Assuming very old.")
        current_date = datetime.min.replace(tzinfo=timezone.utc)

    print(f"Current MLIR-AIE LLVM commit: {current_commit[:8]} ({current_date})")

    target_commit = None
    target_date = None
    reason = ""

    if args.llvm_hash:
        print(f"Using provided LLVM hash: {args.llvm_hash}")
        target_commit = args.llvm_hash
        target_date = get_commit_date(target_commit)
        if not target_date:
            print("Could not get date for provided hash. Aborting.")
            sys.exit(1)
        reason = "Manual update via --llvm-hash"
    else:
        # 2. Get upstream info
        triton_commit = get_triton_llvm_commit()
        triton_date = get_commit_date(triton_commit)
        if triton_commit:
            print(f"Triton LLVM commit:     {triton_commit[:8]} ({triton_date})")
        else:
            print("Failed to fetch Triton LLVM commit.")

        torch_commit = get_torch_mlir_llvm_commit()
        torch_date = get_commit_date(torch_commit)
        if torch_commit:
            print(f"Torch-MLIR LLVM commit: {torch_commit[:8]} ({torch_date})")
        else:
            print("Failed to fetch Torch-MLIR LLVM commit.")

        # 3. Determine target
        potential_targets = []
        if triton_date and triton_date > current_date:
            potential_targets.append((triton_date, triton_commit, "Triton"))
        if torch_date and torch_date > current_date:
            potential_targets.append((torch_date, torch_commit, "Torch-MLIR"))

        if not potential_targets:
            print("No newer LLVM commits found in Triton or Torch-MLIR.")
            return

        # Sort by date descending (newest first)
        potential_targets.sort(key=lambda x: x[0], reverse=True)
        target_date, target_commit, source = potential_targets[0]

        reason = f"Bump to match {source} LLVM {target_commit[:8]} ({target_date})"
        print(f"Found update! {reason}")

    # 4. Find closest eudsl version
    result = find_closest_eudsl_version(target_commit, target_date)

    if not result:
        print("Could not find a suitable eudsl version.")
        sys.exit(1)

    eudsl_llvm_hash, eudsl_llvm_date, new_eudsl_version = result

    print(f"Selected Eudsl: {new_eudsl_version}")
    print(f"  Eudsl's LLVM: {eudsl_llvm_hash[:8]} ({eudsl_llvm_date})")
    print(f"  Target LLVM:  {target_commit[:8]} ({target_date})")

    if target_commit == current_commit:
        print("Target LLVM commit matches current commit. No update needed.")
        return

    # Write reason to GITHUB_OUTPUT if available
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"bump_reason={reason}\n")

    # 5. Look up actual wheel in mlir-distro to get correct DATETIME and version
    wheel_version = None
    wheel_datetime = None
    wheel_info = find_wheel_in_distro(target_commit)
    if wheel_info:
        wheel_version, wheel_datetime = wheel_info
    else:
        print(
            "Wheel not found in mlir-distro. DATETIME will be derived from "
            "commit date (may not match actual wheel) and WHEEL_VERSION "
            "major.minor.patch will not be updated.",
            file=sys.stderr,
        )

    # 6. Update files
    # Use target_commit (from Triton/Torch/User) instead of eudsl's LLVM commit
    update_files(
        target_commit, target_date, new_eudsl_version, wheel_version, wheel_datetime
    )


if __name__ == "__main__":
    main()
