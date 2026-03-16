import os
import sys
import re
import time
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path


def _pip_download_with_retry(cmd, max_attempts=5):
    """Run a pip download command with exponential backoff on failure."""
    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"Running (attempt {attempt}/{max_attempts}): {' '.join(cmd)}",
                file=sys.stderr,
            )
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == max_attempts:
                raise
            wait = 2**attempt
            print(
                f"pip download failed (attempt {attempt}/{max_attempts}), retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)


def install_eudsl(req_file, target_dir):
    print(f"Reading requirements from: {req_file}")
    with open(req_file) as f:
        content = f.read()

    version_match = re.search(r"eudsl-python-extras==(\S+)", content)
    if not version_match:
        print(
            "Could not find eudsl-python-extras version in requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)
    version = version_match.group(1)

    config_match = re.search(r'--config-settings="([^"]+)"', content)
    config_setting = (
        config_match.group(1)
        if config_match
        else "EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie"
    )

    find_links_match = re.search(r"-f\s+(\S+)", content)
    find_links = (
        find_links_match.group(1)
        if find_links_match
        else "https://llvm.github.io/eudsl"
    )

    os.makedirs(target_dir, exist_ok=True)
    print(f"Vendoring eudsl-python-extras=={version} to {target_dir}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as temp_dir:
        download_dir = os.path.join(temp_dir, "download")
        os.makedirs(download_dir)

        # `pip download` resolves the sdist from the find-links index and verifies
        # its hash before saving locally. Retried with backoff since the index links
        # to GitHub release assets, which can return transient 502 errors.
        download_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            f"eudsl-python-extras=={version}",
            "--no-deps",
            "--no-binary",
            "eudsl-python-extras",
            "--no-cache-dir",
            "-f",
            find_links,
            "-d",
            download_dir,
        ]
        _pip_download_with_retry(download_cmd)

        sdists = list(Path(download_dir).glob("eudsl_python_extras-*.tar.gz"))
        if len(sdists) != 1:
            print(
                f"ERROR: expected 1 sdist in {download_dir}, found: {sdists}",
                file=sys.stderr,
            )
            sys.exit(1)
        sdist_path = sdists[0]

        install_dir = os.path.join(temp_dir, "install")
        os.makedirs(install_dir)

        # --no-build-isolation uses build deps from the active environment rather
        # than fetching them from PyPI, keeping this step fully network-free.
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(sdist_path),
            "--target",
            install_dir,
            "--no-deps",
            "--no-build-isolation",
            "--config-settings",
            config_setting,
        ]

        env = os.environ.copy()
        # Pass config_setting as an env var in addition to --config-settings;
        # eudsl's custom build backend reads both forms.
        if "=" in config_setting:
            key, val = config_setting.split("=", 1)
            env[key] = val

        subprocess.run(install_cmd, check=True, env=env)

        shutil.copytree(install_dir, target_dir, dirs_exist_ok=True)

    # Derive the installed prefix from config_setting for verification.
    prefix = config_setting.split("=", 1)[1] if "=" in config_setting else "mlir"
    util_path = os.path.join(target_dir, prefix, "extras", "util.py")
    if not os.path.exists(util_path):
        print(f"ERROR: {util_path} not found!", file=sys.stderr)
        sys.exit(1)

    # Patch np.bool -> np.bool_ (removed in NumPy 1.24).
    with open(util_path) as f:
        util_content = f.read()
    if "np.bool:" in util_content:
        util_content = util_content.replace("np.bool:", "np.bool_:")
        with open(util_path, "w") as f:
            f.write(util_content)
        print(f"Patched np.bool -> np.bool_ in {util_path}", file=sys.stderr)


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
