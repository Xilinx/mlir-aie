#!/usr/bin/env python3

import argparse
import re
import sys
import json
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# Constants
TRITON_LLVM_HASH_URL = (
    "https://raw.githubusercontent.com/triton-lang/triton/main/cmake/llvm-hash.txt"
)
TRITON_HISTORY_URL = "https://api.github.com/repos/triton-lang/triton/commits?path=cmake/llvm-hash.txt&since={}"
TRITON_FILE_URL = (
    "https://raw.githubusercontent.com/triton-lang/triton/{}/cmake/llvm-hash.txt"
)

TORCH_MLIR_SUBMODULE_URL = (
    "https://api.github.com/repos/llvm/torch-mlir/contents/externals/llvm-project"
)
TORCH_MLIR_HISTORY_URL = "https://api.github.com/repos/llvm/torch-mlir/commits?path=externals/llvm-project&since={}"
TORCH_MLIR_FILE_URL = "https://api.github.com/repos/llvm/torch-mlir/contents/externals/llvm-project?ref={}"

LLVM_PROJECT_COMMIT_URL = "https://api.github.com/repos/llvm/llvm-project/commits/{}"
EUDSL_INDEX_URL = "https://llvm.github.io/eudsl/"
EUDSL_SUBMODULE_URL = (
    "https://api.github.com/repos/llvm/eudsl/contents/third_party/llvm-project?ref={}"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONE_LLVM_SH = REPO_ROOT / "utils" / "clone-llvm.sh"
REQUIREMENTS_TXT = REPO_ROOT / "python" / "requirements.txt"


def get_request(url):
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req)


def get_current_llvm_commit():
    content = CLONE_LLVM_SH.read_text()
    match = re.search(r"LLVM_PROJECT_COMMIT=([a-f0-9]+)", content)
    if match:
        return match.group(1)
    return None


def get_triton_llvm_commit():
    try:
        with get_request(TRITON_LLVM_HASH_URL) as response:
            content = response.read().decode("utf-8").strip()
            # The file contains just the hash, maybe with newline
            if re.match(r"^[a-f0-9]+$", content):
                return content
    except Exception as e:
        print(f"Error fetching Triton LLVM commit: {e}", file=sys.stderr)
    return None


def get_torch_mlir_llvm_commit():
    try:
        with get_request(TORCH_MLIR_SUBMODULE_URL) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["sha"]
    except Exception as e:
        print(f"Error fetching Torch-MLIR LLVM commit: {e}", file=sys.stderr)
    return None


def get_since_date():
    return (datetime.now() - timedelta(days=30)).isoformat()


def get_triton_history():
    # Returns list of (llvm_hash, commit_date)
    history = []
    since = get_since_date()
    try:
        url = TRITON_HISTORY_URL.format(since)
        with get_request(url) as response:
            commits = json.loads(response.read().decode("utf-8"))
            for commit in commits:
                sha = commit["sha"]
                commit_date = commit["commit"]["committer"][
                    "date"
                ]  # Triton repo commit date
                try:
                    file_url = TRITON_FILE_URL.format(sha)
                    with get_request(file_url) as file_response:
                        content = file_response.read().decode("utf-8").strip()
                        if re.match(r"^[a-f0-9]+$", content):
                            history.append((content, commit_date))
                except Exception as e:
                    print(f"Error fetching Triton file at {sha}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error fetching Triton history: {e}", file=sys.stderr)
    return history


def get_torch_mlir_history():
    # Returns list of (llvm_hash, commit_date)
    history = []
    since = get_since_date()
    try:
        url = TORCH_MLIR_HISTORY_URL.format(since)
        with get_request(url) as response:
            commits = json.loads(response.read().decode("utf-8"))
            for commit in commits:
                sha = commit["sha"]
                commit_date = commit["commit"]["committer"][
                    "date"
                ]  # Torch-MLIR repo commit date
                try:
                    file_url = TORCH_MLIR_FILE_URL.format(sha)
                    with get_request(file_url) as file_response:
                        data = json.loads(file_response.read().decode("utf-8"))
                        history.append((data["sha"], commit_date))
                except Exception as e:
                    print(
                        f"Error fetching Torch-MLIR file at {sha}: {e}", file=sys.stderr
                    )
    except Exception as e:
        print(f"Error fetching Torch-MLIR history: {e}", file=sys.stderr)
    return history


def get_eudsl_llvm_map():
    """
    Fetches eudsl package versions, then for each version, fetches the LLVM commit hash it uses.
    Returns a map of {llvm_full_hash: eudsl_version_string}.
    """
    eudsl_versions = []  # List of (date_str, eudsl_hash, full_version)
    try:
        with urllib.request.urlopen(EUDSL_INDEX_URL) as response:
            content = response.read().decode("utf-8")
            # Regex: eudsl_python_extras-0\.1\.0\.(\d+\.\d+)\+([a-f0-9]+)\.tar\.gz
            matches = re.findall(
                r"eudsl_python_extras-0\.1\.0\.(\d+\.\d+)\+([a-f0-9]+)\.tar\.gz",
                content,
            )
            for date_str, eudsl_hash in matches:
                full_version = f"0.1.0.{date_str}+{eudsl_hash}"
                eudsl_versions.append((date_str, eudsl_hash, full_version))
    except Exception as e:
        print(f"Error fetching eudsl versions: {e}", file=sys.stderr)
        return {}

    # Sort by date descending to check latest first
    eudsl_versions.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by eudsl_hash and filter by date (last 30 days)
    unique_eudsl_versions = []
    seen_hashes = set()

    # Current date for comparison
    now = datetime.now()

    for date_str, eudsl_hash, full_version in eudsl_versions:
        # date_str is YYYYMMDD.HHMM
        try:
            version_date = datetime.strptime(date_str.split(".")[0], "%Y%m%d")
            days_diff = (now - version_date).days
            if days_diff > 30:
                continue
        except ValueError:
            continue

        if eudsl_hash not in seen_hashes:
            unique_eudsl_versions.append((date_str, eudsl_hash, full_version))
            seen_hashes.add(eudsl_hash)

    llvm_map = {}
    # Check all unique versions from the last month
    print(
        f"Checking {len(unique_eudsl_versions)} eudsl versions from the last 30 days..."
    )
    for _, eudsl_hash, full_version in unique_eudsl_versions:
        try:
            url = EUDSL_SUBMODULE_URL.format(eudsl_hash)
            with get_request(url) as response:
                data = json.loads(response.read().decode("utf-8"))
                llvm_hash = data["sha"]
                llvm_map[llvm_hash] = full_version
                # Also store short hash for convenience if needed, but full hash is safer
                llvm_map[llvm_hash[:7]] = full_version
        except Exception as e:
            print(
                f"Error fetching LLVM hash for eudsl commit {eudsl_hash}: {e}",
                file=sys.stderr,
            )

    return llvm_map


def get_commit_date(commit_hash):
    url = LLVM_PROJECT_COMMIT_URL.format(commit_hash)
    try:
        with get_request(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            # Date format: 2025-12-05T18:00:00Z
            date_str = data["commit"]["committer"]["date"]
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error fetching commit date for {commit_hash}: {e}", file=sys.stderr)
        return None


def update_files(new_commit, commit_date, eudsl_version):
    # Format dates
    # DATETIME=YYYYMMDDHH
    datetime_str = commit_date.strftime("%Y%m%d%H")

    print(f"Updating to LLVM commit: {new_commit}")
    print(f"Commit Date: {commit_date}")
    print(f"DATETIME: {datetime_str}")
    print(f"EUDSL Version: {eudsl_version}")

    # Update utils/clone-llvm.sh
    content = CLONE_LLVM_SH.read_text()
    content = re.sub(
        r"LLVM_PROJECT_COMMIT=[a-f0-9]+", f"LLVM_PROJECT_COMMIT={new_commit}", content
    )
    content = re.sub(r"DATETIME=\d+", f"DATETIME={datetime_str}", content)
    CLONE_LLVM_SH.write_text(content)
    print(f"Updated {CLONE_LLVM_SH}")

    # Update python/requirements.txt
    content = REQUIREMENTS_TXT.read_text()
    # Regex to match eudsl-python-extras==...
    # eudsl-python-extras==0.1.0.20251215.1715+3c7ac1b \
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


def main():
    parser = argparse.ArgumentParser(
        description="Update LLVM version based on Triton or Torch-MLIR."
    )
    parser.add_argument("--llvm-hash", help="Specific LLVM hash to use.")
    args = parser.parse_args()

    current_commit = get_current_llvm_commit()
    print(f"Current LLVM commit: {current_commit}")

    new_commit = None

    llvm_map = get_eudsl_llvm_map()
    print(f"Mapped {len(llvm_map)} eudsl versions to LLVM commits.")

    if args.llvm_hash:
        print(f"Using provided LLVM hash: {args.llvm_hash}")
        new_commit = args.llvm_hash
    else:
        triton_history = get_triton_history()
        torch_mlir_history = get_torch_mlir_history()

        triton_hashes = {h[0] for h in triton_history}
        torch_mlir_hashes = {h[0] for h in torch_mlir_history}

        print(f"Found {len(triton_hashes)} unique Triton LLVM commits.")
        print(f"Found {len(torch_mlir_hashes)} unique Torch-MLIR LLVM commits.")

        current_date = get_commit_date(current_commit)
        if not current_date:
            print("Could not get date for current commit. Assuming very old.")
            current_date = datetime.min

        print(f"Current MLIR-AIE LLVM commit: {current_commit} ({current_date})")

        print("\n--- Comparison ---")
        print("EUDSL Versions (Last 30 days):")
        # Reconstruct eudsl list for printing
        # We need to iterate llvm_map but it's not sorted by date.
        # We can sort by eudsl version string date.
        sorted_eudsl = sorted(
            llvm_map.items(), key=lambda x: x[1].split(".")[3], reverse=True
        )
        for llvm_hash, eudsl_version in sorted_eudsl:
            if len(llvm_hash) < 40:
                continue
            # Extract date from version string: 0.1.0.YYYYMMDD.HHMM+...
            try:
                date_part = eudsl_version.split(".")[3]
                time_part = eudsl_version.split(".")[4].split("+")[0]
                eudsl_date = f"{date_part} {time_part}"
            except:
                eudsl_date = "Unknown"

            status = []
            if llvm_hash in triton_hashes:
                status.append("In Triton")
            if llvm_hash in torch_mlir_hashes:
                status.append("In Torch-MLIR")
            if llvm_hash == current_commit:
                status.append("Current")

            status_str = f" [{', '.join(status)}]" if status else ""
            print(
                f"  {eudsl_version} -> LLVM: {llvm_hash[:8]}... ({eudsl_date}){status_str}"
            )

        print("\nTriton History (Last 30 days):")
        for h, d in triton_history:
            print(f"  LLVM: {h[:8]}... (Triton commit date: {d})")

        print("\nTorch-MLIR History (Last 30 days):")
        for h, d in torch_mlir_history:
            print(f"  LLVM: {h[:8]}... (Torch-MLIR commit date: {d})")
        print("------------------\n")

        candidates = []
        for llvm_hash, eudsl_version in llvm_map.items():
            if len(llvm_hash) < 40:
                continue  # Skip short hashes in map

            commit_date = get_commit_date(llvm_hash)
            if commit_date and commit_date > current_date:
                candidates.append((commit_date, llvm_hash, eudsl_version))

        # Sort by date ascending (oldest to newest)
        candidates.sort(key=lambda x: x[0])

        matches = []
        for c in candidates:
            llvm_hash = c[1]
            if llvm_hash in triton_hashes or llvm_hash in torch_mlir_hashes:
                matches.append(c)

        reason = ""
        if matches:
            # Pick the newest match
            best_commit = matches[-1]
            new_commit = best_commit[1]
            eudsl_version = best_commit[2]

            sources = []
            if new_commit in triton_hashes:
                sources.append("Triton")
            if new_commit in torch_mlir_hashes:
                sources.append("Torch-MLIR")
            reason = (
                f"Bump to LLVM {new_commit[:8]} (matched with {', '.join(sources)})"
            )
            print(f"Found update! {reason}")
        elif candidates:
            # Pick the oldest candidate (next available)
            best_commit = candidates[0]
            new_commit = best_commit[1]
            eudsl_version = best_commit[2]
            reason = f"Bump to next available Eudsl version {eudsl_version} (LLVM {new_commit[:8]}). No match with Triton/Torch-MLIR history found."
            print(f"Found update! {reason}")
        else:
            print("No newer Eudsl versions found.")
            return

        # Write reason to GITHUB_OUTPUT if available
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"bump_reason={reason}\n")

    if new_commit == current_commit:
        print("Already on the target commit.")
        return

    # Get eudsl version
    if new_commit in llvm_map:
        eudsl_version = llvm_map[new_commit]
    elif new_commit[:7] in llvm_map:
        eudsl_version = llvm_map[new_commit[:7]]
    else:
        print(
            f"Warning: Commit {new_commit} not found in eudsl packages. Cannot determine eudsl version."
        )
        print("Aborting.")
        sys.exit(1)

    commit_date = get_commit_date(new_commit)
    if not commit_date:
        print("Failed to get commit date. Aborting.")
        sys.exit(1)

    update_files(new_commit, commit_date, eudsl_version)


if __name__ == "__main__":
    main()
