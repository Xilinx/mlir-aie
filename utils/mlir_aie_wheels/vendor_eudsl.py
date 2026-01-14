import os
import sys
import re
import subprocess
import argparse
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

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"eudsl-python-extras=={version}",
        "--target",
        str(target_dir),
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
    if "=" in config_setting:
        key, val = config_setting.split("=", 1)
        env[key] = val

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


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
