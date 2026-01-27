import os
import sys
import re
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path


def install_eudsl(req_file, target_dir):
    print(f"Reading requirements from: {req_file}")
    with open(req_file) as f:
        content = f.read()

    # Extract the version and config settings from the file content
    version_match = re.search(r"eudsl-python-extras==(\S+)", content)
    if not version_match:
        print(
            "Could not find eudsl-python-extras version in requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    version = version_match.group(1)

    # Extract config settings
    config_match = re.search(r'--config-settings="([^"]+)"', content)
    config_setting = (
        config_match.group(1)
        if config_match
        else "EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie"
    )

    # Extract find-links
    find_links_match = re.search(r"-f\s+(\S+)", content)
    find_links = (
        find_links_match.group(1)
        if find_links_match
        else "https://llvm.github.io/eudsl"
    )

    # Ensure target dir exists
    os.makedirs(target_dir, exist_ok=True)

    print(f"Vendoring eudsl-python-extras=={version} to {target_dir}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"eudsl-python-extras=={version}",
            "--target",
            temp_dir,
            "--no-deps",
            "--no-binary",
            "eudsl-python-extras",
            "--no-cache-dir",
            "--config-settings",
            config_setting,
            "-f",
            find_links,
        ]

        env = os.environ.copy()
        # Apply the same setting both via pip's --config-settings flag (see cmd above)
        # and as an environment variable. Some build backends or tooling may rely on
        # the environment variable form rather than reading --config-settings, so we
        # intentionally support both here for compatibility.
        if "=" in config_setting:
            key, val = config_setting.split("=", 1)
            env[key] = val

        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, env=env)

        print(f"Copying files from {temp_dir} to {target_dir}", file=sys.stderr)
        shutil.copytree(temp_dir, target_dir, dirs_exist_ok=True)

    print(f"Listing files in {target_dir}:", file=sys.stderr)
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            print(os.path.join(root, file), file=sys.stderr)

    # Verify util.py exists
    util_path = os.path.join(target_dir, "aie", "extras", "util.py")
    if not os.path.exists(util_path):
        print(f"ERROR: {util_path} not found!", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Vendor eudsl-python-extras")
    parser.add_argument(
        "--requirements", required=True, help="Path to requirements.txt"
    )
    parser.add_argument(
        "--target", required=True, help="Target directory for installation"
    )
    args = parser.parse_args()

    install_eudsl(args.requirements, args.target)


if __name__ == "__main__":
    main()
