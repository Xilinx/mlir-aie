#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

##===------ iron_setup.py - Cross-platform IRON environment setup -----------===##
#
# Create or reconcile the wheel-backed Python environment used by this checkout.
# Tagged checkouts use their matching release wheel. --dev installs mlir_aie
# from the latest rolling development channel with pip --upgrade --pre, plus
# contributor dependencies.
#
#   python utils/iron_setup.py
#   python utils/iron_setup.py --dev --extras
#
# Setup writes local activation helpers in the repository root:
#
#   Linux / WSL :   source ./iron_env.sh
#   PowerShell  :   . ./iron_env.ps1
#   cmd.exe     :   call .\iron_env.cmd
#
##===----------------------------------------------------------------------===##

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IS_WINDOWS = (os.name == "nt") or sys.platform.startswith("win")
IS_WSL = bool(os.environ.get("WSL_DISTRO_NAME"))
WHEEL_ASSET_INDEX = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets"
# Tagged checkouts use matching release assets; development checkouts use this channel.
ROLLING_WHEEL_CHANNEL = "latest-wheels-4"


# --------------------------------------------------------------------------------------
# Shell emission helpers
# --------------------------------------------------------------------------------------


def sh_quote(value: str) -> str:
    """Quote a literal for a POSIX shell."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ps_quote(value: str) -> str:
    """Quote a literal for PowerShell without variable interpolation."""
    return "'" + value.replace("'", "''") + "'"


def cmd_quote(value: str) -> str:
    """Quote a path passed to cmd.exe."""
    return f'"{value}"'


def emit_set(shell: str, key: str, value: str) -> list[str]:
    if shell == "pwsh":
        return [f"$env:{key} = {ps_quote(value)}"]
    if shell == "cmd":
        return [f'set "{key}={value}"']
    return [f"export {key}={sh_quote(value)}"]


def emit_prepend_path(shell: str, var_name: str, prefix: str) -> list[str]:
    """Prepend a known tool path for the generated activation helper."""
    if shell == "pwsh":
        return [
            f"if ($env:{var_name}) {{ $env:{var_name} = {ps_quote(prefix)} + ';' + $env:{var_name} }}",
            f"else {{ $env:{var_name} = {ps_quote(prefix)} }}",
        ]
    if shell == "cmd":
        return [
            f'if defined {var_name} (set "{var_name}={prefix};%{var_name}%") else (set "{var_name}={prefix}")'
        ]
    return [f'export {var_name}={sh_quote(prefix)}"${{{var_name}:+:${{{var_name}}}}}"']


def emit_prepend_paths(shell: str, var_name: str, paths: list[Path]) -> list[str]:
    """Prepend existing paths while preserving the caller's order."""
    lines: list[str] = []
    # Each emitted statement prepends, so emit paths in reverse order.
    for path in reversed(paths):
        if path.is_dir():
            lines.extend(emit_prepend_path(shell, var_name, str(path)))
    return lines


# --------------------------------------------------------------------------------------
# Subprocess and virtual-environment helpers
# --------------------------------------------------------------------------------------


def _format_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if IS_WINDOWS else shlex.join(cmd)


class CommandError(RuntimeError):
    pass


def run_checked(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> Optional[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=capture,
    )
    if proc.returncode != 0:
        output = str(proc.stdout or "") if capture else ""
        message = f"Command failed ({proc.returncode}): {_format_cmd(cmd)}"
        if output:
            message += "\n" + output
        raise CommandError(message)
    return str(proc.stdout) if capture else None


def capture_text(cmd: list[str], *, cwd: Optional[Path] = None) -> str:
    return run_checked(cmd, cwd=cwd, capture=True) or ""


@dataclass(frozen=True)
class VenvInfo:
    venv_dir: Path
    python: Path


def venv_python_path(venv_dir: Path) -> Path:
    # Virtual environments use Scripts/ on Windows and bin/ on POSIX hosts.
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path, *, python_exe: str) -> VenvInfo:
    if not venv_dir.exists():
        run_checked([python_exe, "-m", "venv", str(venv_dir)])
    python = venv_python_path(venv_dir)
    if not python.exists():
        raise RuntimeError(f"venv Python executable not found: {python}")
    return VenvInfo(venv_dir=venv_dir, python=python)


def pip_install(venv: VenvInfo, args: list[str]) -> None:
    run_checked([str(venv.python), "-m", "pip", *args])


def pip_install_requirements(
    venv: VenvInfo,
    requirements: Path,
    *,
    force_reinstall: bool,
    require_hashes: bool = False,
) -> None:
    args = ["install"]
    if force_reinstall:
        args.append("--force-reinstall")
    if require_hashes:
        args.append("--require-hashes")
    args += ["-r", str(requirements)]
    pip_install(venv, args)


def pip_install_package(
    venv: VenvInfo,
    package: str,
    *,
    force_reinstall: bool,
    upgrade: bool = False,
    allow_prereleases: bool = False,
    find_links: Optional[str] = None,
    wheelhouse: Optional[Path] = None,
    no_index: bool = False,
    no_deps: bool = False,
) -> None:
    args = ["install"]
    if upgrade:
        args.append("--upgrade")
    if allow_prereleases:
        args.append("--pre")
    if force_reinstall:
        args.append("--force-reinstall")
    if no_deps:
        args.append("--no-deps")
    if no_index:
        args.append("--no-index")
    if wheelhouse is not None:
        args += ["--find-links", str(wheelhouse)]
    if find_links:
        args += ["--find-links", find_links]
    args.append(package)
    pip_install(venv, args)


def installed_prefix(
    venv: VenvInfo,
    dist_name: str,
    candidates: list[str],
    *,
    require_subdir: Optional[str] = None,
) -> Optional[Path]:
    """Find a package-owned install prefix below pip's site-packages directory."""
    try:
        output = capture_text([str(venv.python), "-m", "pip", "show", dist_name])
    except CommandError:
        return None

    location = next(
        (
            line.split(":", 1)[1].strip()
            for line in output.replace("\r", "").splitlines()
            if line.lower().startswith("location:")
        ),
        "",
    )
    if not location:
        return None

    base = Path(location).resolve()
    for candidate in candidates:
        prefix = (base / candidate).resolve()
        if not prefix.is_dir():
            continue
        if require_subdir is None or (prefix / require_subdir).is_dir():
            return prefix
    return None


# --------------------------------------------------------------------------------------
# Published wheel resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelSelection:
    package: str
    find_links: Optional[str]
    wheelhouse: Optional[Path]
    description: str
    upgrade: bool = False
    allow_prereleases: bool = False


def release_tag_at_head(repo_root: Path) -> Optional[str]:
    """Return a release-like tag only when it names the current commit."""
    if not (repo_root / ".git").exists() or not shutil.which("git"):
        return None
    try:
        tag = capture_text(
            ["git", "describe", "--exact-match", "--tags", "HEAD"], cwd=repo_root
        ).strip()
    except CommandError:
        return None
    return tag if re.fullmatch(r"v\d[0-9A-Za-z.+-]*", tag) else None


def resolve_mlir_aie_wheel(args: argparse.Namespace, repo_root: Path) -> WheelSelection:
    """Choose a local, matching-release, or rolling mlir_aie wheel source."""
    if args.wheelhouse:
        wheelhouse = Path(args.wheelhouse).expanduser()
        if not wheelhouse.is_absolute():
            wheelhouse = (repo_root / wheelhouse).resolve()
        if not wheelhouse.is_dir():
            raise RuntimeError(f"Wheelhouse directory not found: {wheelhouse}")
        # A supplied wheelhouse is authoritative and must stay offline.
        return WheelSelection(
            package="mlir_aie",
            find_links=None,
            wheelhouse=wheelhouse,
            description=f"local wheelhouse: {wheelhouse}",
        )

    if args.dev:
        # Contributor environments must track the newest rolling prerelease.
        # pip needs both --upgrade and --pre to make that guarantee.
        return WheelSelection(
            package="mlir_aie",
            find_links=f"{WHEEL_ASSET_INDEX}/{ROLLING_WHEEL_CHANNEL}",
            wheelhouse=None,
            description=(
                "latest rolling development wheel: "
                f"{ROLLING_WHEEL_CHANNEL} (pip --upgrade --pre)"
            ),
            upgrade=True,
            allow_prereleases=True,
        )

    if tag := release_tag_at_head(repo_root):
        # A tagged checkout must use the wheel assets published for that tag.
        return WheelSelection(
            package=f"mlir_aie=={tag.removeprefix('v')}",
            find_links=f"{WHEEL_ASSET_INDEX}/{tag}",
            wheelhouse=None,
            description=f"release tag: {tag}",
        )

    # Untagged non-development checkouts use pip's normal rolling-channel selection.
    return WheelSelection(
        package="mlir_aie",
        find_links=f"{WHEEL_ASSET_INDEX}/{ROLLING_WHEEL_CHANNEL}",
        wheelhouse=None,
        description=f"rolling channel: {ROLLING_WHEEL_CHANNEL}",
        upgrade=True,
    )


def install_mlir_aie(
    venv: VenvInfo,
    selection: WheelSelection,
    *,
    force_reinstall: bool,
) -> None:
    print(f"\nInstalling mlir_aie ({selection.description})...")
    if selection.wheelhouse is not None:
        # Runtime requirements are installed separately; do not consult an index here.
        pip_install_package(
            venv,
            selection.package,
            force_reinstall=True,
            wheelhouse=selection.wheelhouse,
            no_index=True,
            no_deps=True,
        )
        return

    pip_install_package(
        venv,
        selection.package,
        force_reinstall=force_reinstall,
        upgrade=selection.upgrade,
        allow_prereleases=selection.allow_prereleases,
        find_links=selection.find_links,
    )


# --------------------------------------------------------------------------------------
# Windows llvm-aie wheel repair
# --------------------------------------------------------------------------------------


def fixup_llvm_aie_windows(peano_root: Path) -> None:
    """Repair required files in published Windows llvm-aie wheels."""
    if not IS_WINDOWS:
        return

    toolchain_root = peano_root / "lib"
    toolchains = sorted(
        path for path in toolchain_root.glob("*-none-unknown-elf") if path.is_dir()
    )
    if not toolchains:
        raise RuntimeError(f"llvm-aie toolchains not found under: {toolchain_root}")

    objcopy = peano_root / "bin" / "llvm-objcopy.exe"
    objcopy_exe = str(objcopy) if objcopy.exists() else shutil.which("llvm-objcopy.exe")
    if not objcopy_exe:
        raise RuntimeError("llvm-objcopy.exe is required to prepare the llvm-aie wheel")

    for toolchain in toolchains:
        # The published wheel uses .lib names; the AIE linker expects GNU-style aliases.
        for source_name, alias_name in (("c.lib", "libc.a"), ("m.lib", "libm.a")):
            source = toolchain / source_name
            alias = toolchain / alias_name
            if alias.exists():
                continue
            if not source.exists():
                raise RuntimeError(
                    f"llvm-aie is missing both {source_name} and {alias_name} in {toolchain}"
                )
            try:
                shutil.copy2(source, alias)
            except OSError as error:
                raise RuntimeError(
                    f"Failed to create {alias} from {source}: {error}"
                ) from error

        crt1 = toolchain / "crt1.o"
        if not crt1.exists():
            raise RuntimeError(f"llvm-aie is missing {crt1}")

        # Avoid link requests for unavailable MSVC runtime libraries.
        result = subprocess.run(
            [objcopy_exe, "--remove-section=.deplibs", str(crt1)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip()
            raise RuntimeError(
                f"Failed to prepare {crt1} with llvm-objcopy"
                + (f": {detail}" if detail else "")
            )


# --------------------------------------------------------------------------------------
# XRT discovery and Windows pyxrt compatibility reporting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class XrtLayout:
    """XRT paths and compatibility information needed by setup and activation."""

    root: Optional[Path]
    setup_script: Optional[Path]
    pyxrt: Optional[Path]
    pyxrt_abi: Optional[tuple[int, int]]
    warnings: tuple[str, ...]


def _candidate_xrt_root(args: argparse.Namespace) -> Optional[Path]:
    """Return the selected XRT root when one is available."""
    requested = (args.xrt_root or "").strip()
    if requested:
        root = Path(requested).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"Configured XRT root does not exist: {root}")
        return root

    # Native Windows uses XRT_ROOT. Ignore XILINX_XRT so stale Linux settings do not win.
    variables = ("XRT_ROOT",) if IS_WINDOWS else ("XRT_ROOT", "XILINX_XRT")
    for variable in variables:
        value = os.environ.get(variable, "").strip()
        if value:
            root = Path(value).expanduser()
            if root.is_dir():
                return root.resolve()

    # Only use the conventional location when it exists; never emit a fake XRT root.
    default_root = Path(r"C:/Xilinx/XRT") if IS_WINDOWS else Path("/opt/xilinx/xrt")
    return default_root if default_root.is_dir() else None


def _find_windows_pyxrt(root: Optional[Path]) -> Optional[Path]:
    """Locate the Python binding shipped with the selected Windows XRT SDK."""
    if root is None:
        return None
    python_dir = root / "python"
    for candidate in sorted(python_dir.glob("pyxrt*.pyd")):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _windows_pyxrt_abi(path: Path) -> Optional[tuple[int, int]]:
    """Read the CPython ABI imported by a Windows pyxrt extension."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if match := re.search(rb"python(\d)(\d{2})\.dll", data, re.IGNORECASE):
        return int(match.group(1)), int(match.group(2))
    return None


def resolve_xrt_layout(args: argparse.Namespace, venv: VenvInfo) -> XrtLayout:
    """Resolve XRT paths and collect Windows-only pyxrt compatibility warnings."""
    root = _candidate_xrt_root(args)
    setup_script = root / "setup.sh" if root and (root / "setup.sh").is_file() else None
    warnings: list[str] = []

    pyxrt: Optional[Path] = None
    pyxrt_abi: Optional[tuple[int, int]] = None
    if IS_WINDOWS:
        pyxrt = _find_windows_pyxrt(root)
        if root is None:
            warnings.append(
                "No Windows XRT SDK root was found. Install XRT with pyxrt.pyd or "
                "rerun setup with --xrt-root <path>."
            )
        elif pyxrt is None:
            warnings.append(
                f"No pyxrt.pyd was found under {root / 'python'}. Select an XRT SDK "
                "that includes Python bindings."
            )
        else:
            # Warn during setup instead of failing later when native Python JIT imports pyxrt.
            pyxrt_abi = _windows_pyxrt_abi(pyxrt)
            actual = capture_text(
                [
                    str(venv.python),
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ]
            ).strip()
            if pyxrt_abi is None:
                warnings.append(
                    f"Could not determine the CPython ABI used by XRT pyxrt at {pyxrt}. "
                    "Use the documented XRT/Python pairing before running native Python JIT."
                )
            else:
                expected_text = f"{pyxrt_abi[0]}.{pyxrt_abi[1]}"
                if actual != expected_text:
                    warnings.append(
                        f"XRT pyxrt targets CPython {expected_text}, but this environment "
                        f"uses CPython {actual}. Recreate the environment with Python "
                        f"{expected_text}, or select an XRT SDK with matching bindings."
                    )

    return XrtLayout(
        root=root,
        setup_script=setup_script,
        pyxrt=pyxrt,
        pyxrt_abi=pyxrt_abi,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------------------
# NPU detection
# --------------------------------------------------------------------------------------


# NPU_REGEX recognizes supported devices; NPU2_REGEX selects the Makefile NPU2=1 path.
NPU_REGEX = re.compile(
    r"NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]",
    re.IGNORECASE,
)
NPU2_REGEX = re.compile(
    r"NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]",
    re.IGNORECASE,
)

# Windows and WSL use the driver-owned tool so detection matches the installed driver.
WINDOWS_DRIVER_XRT_SMI_DIR = r"C:\Windows\System32\AMD"
WINDOWS_DRIVER_XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
WSL_DRIVER_XRT_SMI = "/mnt/c/Windows/System32/AMD/xrt-smi.exe"


def xrt_smi_commands(xrt: XrtLayout) -> list[list[str]]:
    """Return xrt-smi probe commands in platform-specific precedence order."""
    if IS_WINDOWS:
        # Do not fall back to an XRT copy: the driver tool identifies the installed NPU.
        return [[WINDOWS_DRIVER_XRT_SMI, "examine"]]
    if IS_WSL:
        # WSL probes the Windows driver directly; its Linux XRT package may not match it.
        return [[WSL_DRIVER_XRT_SMI, "examine"]]

    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(command: list[str]) -> None:
        # XRT_ROOT/bin and PATH can resolve to the same executable.
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            commands.append(command)

    # Prefer the explicitly selected XRT installation on native Linux.
    if xrt.root:
        for executable in (
            xrt.root / "bin" / "xrt-smi",
            xrt.root / "bin" / "xrt-smi.exe",
        ):
            if executable.is_file():
                add([str(executable), "examine"])

    # Otherwise use the native XRT installation already visible to the shell.
    for executable in ("xrt-smi", "xrt-smi.exe"):
        if resolved := shutil.which(executable):
            add([resolved, "examine"])

    return commands


def detect_npu2_flag(xrt: XrtLayout) -> tuple[Optional[bool], str]:
    """Detect the Makefile NPU2 selector without making setup depend on it."""
    attempted = False
    for command in xrt_smi_commands(xrt):
        try:
            output = capture_text(command).replace("\r", "")
        except (CommandError, OSError):
            continue
        attempted = True
        if NPU2_REGEX.search(output):
            return True, f"Detected NPU2 device via: {_format_cmd(command)}"
        if NPU_REGEX.search(output):
            return False, f"Detected NPU1 device via: {_format_cmd(command)}"

    # Unknown output must not be treated as NPU2=0.
    if attempted:
        return None, "WARNING: xrt-smi did not report a supported NPU"
    return None, "WARNING: xrt-smi not available (or no NPU detected)"


# --------------------------------------------------------------------------------------
# Activation helper generation
# --------------------------------------------------------------------------------------


def _activate_venv_lines(shell: str, venv: VenvInfo) -> list[str]:
    """Emit an idempotent virtual-environment activation command."""
    if shell == "pwsh":
        activate = venv.venv_dir / "Scripts" / "Activate.ps1"
        return [
            f"if ($env:VIRTUAL_ENV -ne {ps_quote(str(venv.venv_dir))} -and (Test-Path {ps_quote(str(activate))})) {{ . {ps_quote(str(activate))} }}"
        ]
    if shell == "cmd":
        activate = venv.venv_dir / "Scripts" / "activate.bat"
        return [
            f'if /I not "%VIRTUAL_ENV%"=="{venv.venv_dir}" if exist {cmd_quote(str(activate))} call {cmd_quote(str(activate))}'
        ]
    activate = venv.venv_dir / "bin" / "activate"
    return [
        f'if [ "${{VIRTUAL_ENV:-}}" != {sh_quote(str(venv.venv_dir))} ] && [ -f {sh_quote(str(activate))} ]; then . {sh_quote(str(activate))}; fi'
    ]


def build_env_lines(
    shell: str,
    *,
    venv: VenvInfo,
    mlir_prefix: Path,
    peano_prefix: Path,
    xrt: XrtLayout,
    npu2_flag: Optional[bool],
) -> list[str]:
    """Build one activation helper with its final tool-precedence order."""
    comment = "REM" if shell == "cmd" else "#"
    lines = [
        f"{comment} Generated by utils/iron_setup.py. Re-run setup to refresh this file.",
        *_activate_venv_lines(shell, venv),
    ]

    lines.extend(emit_set(shell, "MLIR_AIE_INSTALL_DIR", str(mlir_prefix)))
    # aie.pth handles Python imports; expose only wheel-provided tools here.
    lines.extend(emit_prepend_paths(shell, "PATH", [mlir_prefix / "bin"]))
    library_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
    lines.extend(emit_prepend_paths(shell, library_var, [mlir_prefix / "lib"]))

    lines.extend(emit_set(shell, "PEANO_INSTALL_DIR", str(peano_prefix)))
    lines.append(
        f"{comment} Keep llvm-aie out of PATH so it does not replace the host compiler."
    )

    if IS_WINDOWS:
        # XILINX_XRT is Linux-only; do not carry it into a native Windows shell.
        if shell == "pwsh":
            lines.append(r"Remove-Item Env:\XILINX_XRT -ErrorAction SilentlyContinue")
        elif shell == "cmd":
            lines.append('set "XILINX_XRT="')
        else:
            lines.append("unset XILINX_XRT 2>/dev/null || true")

        if xrt.root:
            lines.extend(emit_set(shell, "XRT_ROOT", str(xrt.root)))
            # SDK tools may live at the root; source installs commonly use bin/.
            lines.extend(
                emit_prepend_paths(
                    shell,
                    "PATH",
                    [xrt.root, xrt.root / "bin", xrt.root / "lib"],
                )
            )
    else:
        # setup.sh owns version-specific Linux runtime variables when it is available.
        if xrt.setup_script and shell == "sh":
            lines.append(
                f"if [ -f {sh_quote(str(xrt.setup_script))} ]; then . {sh_quote(str(xrt.setup_script))}; fi"
            )
        if xrt.root:
            lines.extend(emit_set(shell, "XRT_ROOT", str(xrt.root)))
            lines.extend(emit_set(shell, "XILINX_XRT", str(xrt.root)))
            if not xrt.setup_script:
                # Without setup.sh, provide the minimal runtime search paths ourselves.
                lines.extend(emit_prepend_paths(shell, "PATH", [xrt.root / "bin"]))
                lines.extend(
                    emit_prepend_paths(shell, "LD_LIBRARY_PATH", [xrt.root / "lib"])
                )

    if IS_WINDOWS and xrt.pyxrt:
        # The SDK ships pyxrt outside site-packages.
        lines.extend(emit_prepend_paths(shell, "PYTHONPATH", [xrt.pyxrt.parent]))

    if IS_WINDOWS:
        # Add this last: emitted entries prepend, so the driver tool wins over SDK copies.
        lines.extend(emit_prepend_path(shell, "PATH", WINDOWS_DRIVER_XRT_SMI_DIR))

    if npu2_flag is not None:
        # Preserve the existing Makefile selector only after an explicit detection.
        lines.extend(emit_set(shell, "NPU2", "1" if npu2_flag else "0"))

    return lines


def write_activation_scripts(
    repo_root: Path,
    *,
    venv: VenvInfo,
    mlir_prefix: Path,
    peano_prefix: Path,
    xrt: XrtLayout,
    npu2_flag: Optional[bool],
) -> list[Path]:
    """Write activation helpers for the current host platform."""
    requested = ["cmd", "pwsh"] if IS_WINDOWS else ["sh"]
    names = {"cmd": "iron_env.cmd", "pwsh": "iron_env.ps1", "sh": "iron_env.sh"}
    written: list[Path] = []
    for shell in requested:
        destination = repo_root / names[shell]
        lines = build_env_lines(
            shell,
            venv=venv,
            mlir_prefix=mlir_prefix,
            peano_prefix=peano_prefix,
            xrt=xrt,
            npu2_flag=npu2_flag,
        )
        if shell == "cmd":
            lines = [f"@{line}" for line in lines]
        content = "\n".join(lines)
        # Windows helpers keep native line endings for direct cmd.exe/PowerShell use.
        newline = "\r\n" if shell in {"cmd", "pwsh"} else "\n"
        destination.write_text(content + "\n", encoding="utf-8", newline=newline)
        written.append(destination)
    return written


# --------------------------------------------------------------------------------------
# Setup workflow
# --------------------------------------------------------------------------------------


def install_dev_hooks(venv: VenvInfo, repo_root: Path) -> None:
    """Install the repository's pre-commit and pre-push hooks."""
    run_checked(
        [
            str(venv.python),
            "-m",
            "pre_commit",
            "install",
            "--hook-type",
            "pre-commit",
            "--hook-type",
            "pre-push",
        ],
        cwd=repo_root,
    )


def install_extras(venv: VenvInfo, repo_root: Path, *, force_reinstall: bool) -> None:
    """Install optional ML, notebook, and Jupyter-kernel support."""
    for relative in ("python/requirements_ml.txt", "python/requirements_notebook.txt"):
        requirements = repo_root / relative
        if requirements.is_file():
            pip_install_requirements(
                venv, requirements, force_reinstall=force_reinstall
            )
    try:
        run_checked(
            [
                str(venv.python),
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                venv.venv_dir.name,
            ]
        )
    except CommandError:
        print("Warning: could not register this environment as a Jupyter kernel.")


def installed_package_version(venv: VenvInfo, distribution: str) -> Optional[str]:
    try:
        output = capture_text([str(venv.python), "-m", "pip", "show", distribution])
    except CommandError:
        return None
    for line in output.replace("\r", "").splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip() or None
    return None


def print_completion(
    *,
    venv: VenvInfo,
    xrt: XrtLayout,
    npu2_flag: Optional[bool],
    npu_reason: str,
    scripts: list[Path],
) -> None:
    """Print installed versions, XRT status, and activation commands."""
    python_version = capture_text(
        [str(venv.python), "-c", "import sys; print(sys.version.split()[0])"]
    ).strip()
    mlir_version = installed_package_version(venv, "mlir-aie") or "unknown"
    llvm_version = installed_package_version(venv, "llvm-aie") or "unknown"

    warnings = list(xrt.warnings)
    if warnings:
        print(
            "\nIRON setup completed, but native Python JIT is not configured correctly."
        )
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nIRON environment is ready.")

    print(
        f"\n  Python {python_version} | mlir_aie {mlir_version} | "
        f"llvm-aie {llvm_version}"
    )
    if xrt.root:
        print(f"  XRT: {xrt.root}")
    if npu2_flag is None:
        print(f"  NPU2: not set ({npu_reason})")
    else:
        print(f"  NPU2: {1 if npu2_flag else 0}")

    print("\nTo activate the IRON environment, run:")
    for script in scripts:
        if script.suffix == ".sh":
            print(f"  POSIX shell  source ./{script.name}")
        elif script.suffix == ".ps1":
            print(f"  PowerShell   . .\\{script.name}")
        else:
            print(f"  cmd.exe      call .\\{script.name}")


def install_plan(args: argparse.Namespace, repo_root: Path) -> None:
    """Install the environment and regenerate its activation helpers."""
    venv_dir = (repo_root / args.venv).resolve()
    platform = "Windows" if IS_WINDOWS else "WSL" if IS_WSL else "Linux"

    print(f"\nIRON setup ({platform})")
    default_venv = (repo_root / "ironenv").resolve()
    if venv_dir != default_venv:
        print(f"  Environment: {venv_dir}")
    if not venv_dir.exists():
        print("\nCreating the virtual environment...")
    venv = ensure_venv(venv_dir, python_exe=args.python)
    force_reinstall = bool(args.force_reinstall)

    wheel = resolve_mlir_aie_wheel(args, repo_root)

    print("\nInstalling Python dependencies...")
    pip_install(venv, ["install", "--upgrade", "pip"])

    runtime_requirements = repo_root / "python" / "requirements.txt"
    pip_install_requirements(
        venv, runtime_requirements, force_reinstall=force_reinstall
    )

    install_mlir_aie(venv, wheel, force_reinstall=force_reinstall)

    mlir_prefix = installed_prefix(venv, "mlir_aie", ["mlir_aie"])
    if mlir_prefix is None:
        raise RuntimeError("mlir_aie installation did not produce an install prefix")

    peano_requirements = repo_root / "utils" / "peano-requirements.txt"
    print("\nInstalling llvm-aie...")
    if args.llvm_aie == "nightly":
        pip_install_requirements(
            venv, peano_requirements, force_reinstall=force_reinstall
        )
    else:
        pip_install_package(
            venv,
            "llvm-aie",
            force_reinstall=force_reinstall,
            find_links=args.llvm_aie,
        )

    peano_prefix = installed_prefix(
        venv, "llvm-aie", ["llvm-aie", "llvm_aie"], require_subdir="bin"
    )
    if peano_prefix is None:
        raise RuntimeError("llvm-aie installation did not produce an install prefix")
    if IS_WINDOWS:
        print("\nPreparing the Windows llvm-aie toolchains...")
    fixup_llvm_aie_windows(peano_prefix)

    if args.dev:
        print("\nInstalling contributor tools and Git hooks...")
        dev_requirements = repo_root / "python" / "requirements_dev.lock"
        pip_install_requirements(
            venv,
            dev_requirements,
            force_reinstall=force_reinstall,
            require_hashes=True,
        )
        install_dev_hooks(venv, repo_root)

    if args.extras:
        print("\nInstalling optional ML and notebook support...")
        install_extras(venv, repo_root, force_reinstall=force_reinstall)

    xrt = resolve_xrt_layout(args, venv)
    # Detection is informational for setup but preserves the current Makefile contract.
    npu2_flag, npu_reason = detect_npu2_flag(xrt)
    scripts = write_activation_scripts(
        repo_root,
        venv=venv,
        mlir_prefix=mlir_prefix,
        peano_prefix=peano_prefix,
        xrt=xrt,
        npu2_flag=npu2_flag,
    )
    print_completion(
        venv=venv,
        xrt=xrt,
        npu2_flag=npu2_flag,
        npu_reason=npu_reason,
        scripts=scripts,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_install_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create the virtual environment.",
    )
    parser.add_argument(
        "--venv",
        default="ironenv",
        help="Virtual-environment directory relative to the repository root.",
    )
    parser.add_argument(
        "--dev",
        "--developer",
        dest="dev",
        action="store_true",
        help=(
            "Install locked contributor tooling and Git hooks, and install or "
            "upgrade mlir_aie to the latest rolling development wheel using "
            "pip --upgrade --pre unless --wheelhouse is supplied."
        ),
    )
    parser.add_argument(
        "--extras",
        action="store_true",
        help="Install the optional ML and notebook requirements.",
    )
    parser.add_argument(
        "--wheelhouse",
        default="",
        help="Install mlir_aie from this local wheelhouse instead of published wheels.",
    )
    parser.add_argument(
        "--llvm-aie",
        default="nightly",
        help="llvm-aie wheel source: nightly (default) or a custom --find-links URL.",
    )
    parser.add_argument(
        "--xrt-root",
        default="",
        help="XRT installation directory. Defaults to XRT_ROOT or the platform default; Linux also honors XILINX_XRT.",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Pass --force-reinstall to pip.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or reconcile the IRON environment for this checkout."
    )
    add_install_args(parser)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = build_arg_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    args = parser.parse_args(argv)

    if IS_WINDOWS:
        # Native Windows uses XRT_ROOT; do not leak a stale Linux root to subprocesses.
        os.environ.pop("XILINX_XRT", None)

    try:
        install_plan(args, repo_root)
    except (OSError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
