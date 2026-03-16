import os
import sys
import re
import time
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path


def _parse_requirements(req_file):
    """Parse eudsl-specific fields from a requirements.txt.

    Returns (version, find_links, config_setting).
    """
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

    # Handle both quoted and unquoted forms:
    #   --config-settings="KEY=VALUE"
    #   --config-settings=KEY=VALUE
    config_match = re.search(r'--config-settings=(?:"([^"]+)"|(\S+))', content)
    if config_match:
        config_setting = config_match.group(1) or config_match.group(2)
    else:
        config_setting = "EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie"

    find_links_match = re.search(r"-f\s+(\S+)", content)
    find_links = (
        find_links_match.group(1)
        if find_links_match
        else "https://llvm.github.io/eudsl"
    )

    return version, find_links, config_setting


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
    version, find_links, config_setting = _parse_requirements(req_file)

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


def install_non_eudsl_deps(req_file):
    """Install all non-eudsl packages from req_file via a plain pip install.

    Strips the eudsl-python-extras entry (and its find-links line) before
    passing the requirements to pip. Used in test environments where eudsl is
    already vendored inside the installed wheel and does not need to be fetched.
    """
    with open(req_file) as f:
        lines = f.readlines()

    # Strip the -f find-links line for the eudsl index and the
    # eudsl-python-extras entry (which may span multiple lines via backslash
    # continuation) along with its --config-settings continuation line.
    filtered = []
    skip_continuation = False
    for line in lines:
        stripped = line.rstrip()
        if skip_continuation:
            skip_continuation = stripped.endswith("\\")
            continue
        if re.match(r"\s*-f\s+\S*eudsl", stripped):
            continue
        if re.match(r"\s*eudsl-python-extras", stripped):
            skip_continuation = stripped.endswith("\\")
            continue
        filtered.append(line)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.writelines(filtered)
        tmp_path = tmp.name

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", tmp_path],
            check=True,
        )
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Vendor eudsl-python-extras")
    parser.add_argument(
        "--requirements", required=True, help="Path to requirements.txt"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help="Target directory for vendoring installation")
    group.add_argument(
        "--install-non-eudsl",
        action="store_true",
        help=(
            "Install all non-eudsl packages from requirements into the active "
            "environment. Use when eudsl is already vendored inside an installed "
            "wheel and does not need to be fetched."
        ),
    )
    args = parser.parse_args()

    if args.install_non_eudsl:
        install_non_eudsl_deps(args.requirements)
    else:
        install_eudsl(args.requirements, args.target)


if __name__ == "__main__":
    main()
