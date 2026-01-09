#!/usr/bin/env python3

import urllib.request
import re
import json
import os
import sys


def get_latest_eudsl_version():
    url = "https://llvm.github.io/eudsl/"
    print(f"Fetching {url}...")
    with urllib.request.urlopen(url) as response:
        html = response.read().decode("utf-8")

    # Regex to find eudsl-python-extras tarballs
    # Format: eudsl_python_extras-0.1.0.<YYYYMMDD>.<TIME>+<COMMIT>.tar.gz
    # Example: eudsl_python_extras-0.1.0.20260109.1040+c3dfefd.tar.gz
    pattern = re.compile(
        r"eudsl_python_extras-0\.1\.0\.(\d{8})\.(\d+)\+([a-f0-9]+)\.tar\.gz"
    )

    matches = []
    for match in pattern.finditer(html):
        date = match.group(1)
        time = match.group(2)
        commit = match.group(3)
        full_version = f"0.1.0.{date}.{time}+{commit}"
        matches.append(
            {
                "date": date,
                "time": time,
                "commit": commit,
                "version": full_version,
                "filename": match.group(0),
            }
        )

    if not matches:
        raise Exception("No eudsl-python-extras versions found")

    # Sort by date and time (descending)
    matches.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
    latest = matches[0]
    print(f"Latest eudsl-python-extras version: {latest['version']}")
    return latest


def get_llvm_commit(eudsl_commit):
    url = f"https://api.github.com/repos/llvm/eudsl/contents/third_party/llvm-project?ref={eudsl_commit}"
    print(f"Fetching LLVM submodule info from {url}...")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

    llvm_sha = data["sha"]
    print(f"Resolved LLVM commit: {llvm_sha}")
    return llvm_sha


def update_clone_llvm(llvm_sha, date):
    filepath = os.path.join(os.path.dirname(__file__), "clone-llvm.sh")
    with open(filepath, "r") as f:
        content = f.read()

    # Update LLVM_PROJECT_COMMIT
    new_content = re.sub(
        r"LLVM_PROJECT_COMMIT=[a-f0-9]+", f"LLVM_PROJECT_COMMIT={llvm_sha}", content
    )

    # Update DATETIME
    # Pad date with 00 to match YYYYMMDDHH format if needed, or just use what we have if it's compatible.
    # Previous was 2025120518 (10 digits). New date is 20260109 (8 digits).
    # We should pad it to 10 digits to ensure it's numerically larger if we want to maintain that style,
    # although WHEEL_VERSION just concatenates it.
    # Let's pad with 00 (midnight).
    new_datetime = f"{date}00"
    new_content = re.sub(r"DATETIME=\d+", f"DATETIME={new_datetime}", new_content)

    if content != new_content:
        print(f"Updating {filepath}...")
        with open(filepath, "w") as f:
            f.write(new_content)
        return True
    return False


def update_requirements(new_version):
    filepath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "python", "requirements_extras.txt"
    )
    with open(filepath, "r") as f:
        content = f.read()

    # Update eudsl-python-extras version
    new_content = re.sub(
        r"eudsl-python-extras==[\w\.\+\-]+",
        f"eudsl-python-extras=={new_version}",
        content,
    )

    if content != new_content:
        print(f"Updating {filepath}...")
        with open(filepath, "w") as f:
            f.write(new_content)
        return True
    return False


def main():
    try:
        latest_eudsl = get_latest_eudsl_version()
        llvm_sha = get_llvm_commit(latest_eudsl["commit"])

        updated_llvm = update_clone_llvm(llvm_sha, latest_eudsl["date"])
        updated_reqs = update_requirements(latest_eudsl["version"])

        if updated_llvm or updated_reqs:
            print(f"BUMP_REQUIRED=true")
            print(f"NEW_LLVM_SHA={llvm_sha}")
            print(f"NEW_EUDSL_VERSION={latest_eudsl['version']}")

            # Write to GITHUB_OUTPUT if running in Actions
            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"bump_required=true\n")
                    f.write(f"new_llvm_sha={llvm_sha}\n")
                    f.write(f"new_eudsl_version={latest_eudsl['version']}\n")
                    f.write(f"new_date={latest_eudsl['date']}\n")
        else:
            print("No updates required.")
            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"bump_required=false\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
