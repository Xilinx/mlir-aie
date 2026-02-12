#!/usr/bin/env python3
##===------ iron_setup.py - One-stop mlir-aie/Iron setup (Linux/WSL/Windows). -----===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===------------------------------------------------------------------------------===##
#
# This user-friendly script sets up the environment and installs packages for IRON.
# It is cross-platform and works similarly on native Linux, WSL, and native Windows.
#
#   1) First-time install:
#      python3 utils/iron_setup.py (optional: --all)
#
#   2) Update to newest wheels/deps:
#      python3 utils/iron_setup.py update (optional: --all)
#
#   3) Set env vars for a new shell/session:
#
#      * sh/bash    : eval "$(python3 utils/iron_setup.py env)"
#      * PowerShell : python utils/iron_setup.py env --shell pwsh | iex
#      * cmd.exe    : python utils\iron_setup.py env --shell cmd > "%TEMP%\iron_env.bat" && call "%TEMP%\iron_env.bat"
#
##===------------------------------------------------------------------------------===##


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


# --------------------------------------------------------------------------------------
# Shell emission helpers
# --------------------------------------------------------------------------------------


def sh_quote(value: str) -> str:
    # Safe single quotes in POSIX shells: abc'def --> 'abc'"'"'def'
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ps_quote(value: str) -> str:
    # PowerShell double-quote string; escape embedded ` and ".
    escaped = value.replace("`", "``").replace('"', '`"')
    return f'"{escaped}"'


def cmd_quote(value: str) -> str:
    # Simple cmd.exe quoting for paths.
    return f'"{value}"'


def default_env_shell() -> str:
    return "pwsh" if IS_WINDOWS else "sh"


def emit_set(shell: str, key: str, value: str) -> list[str]:
    shell = shell.lower()
    if shell == "pwsh":
        return [f"$env:{key} = {ps_quote(value)}"]
    if shell == "cmd":
        return [f'set "{key}={value}"']
    return [f"export {key}={sh_quote(value)}"]


def emit_prepend_path(shell: str, var_name: str, prefix: str) -> list[str]:
    shell = shell.lower()
    sep = ";" if shell in ("pwsh", "cmd") else ":"
    if shell == "pwsh":
        p = ps_quote(prefix)
        return [
            f"if ($env:{var_name}) {{ $env:{var_name} = {ps_quote(prefix + sep)} + $env:{var_name} }} "
            f"else {{ $env:{var_name} = {p} }}"
        ]
    if shell == "cmd":
        return [
            f'if defined {var_name} (set "{var_name}={prefix}{sep}%{var_name}%") else (set "{var_name}={prefix}")'
        ]
    # 'prefix'${VAR+:$VAR}
    return [f"export {var_name}={sh_quote(prefix)}${{{var_name}:+{sep}${var_name}}}"]


def emit_append_path_if_exists(shell: str, var_name: str, suffix_dir: str) -> list[str]:
    shell = shell.lower()
    sep = ";" if shell in ("pwsh", "cmd") else ":"
    if shell == "pwsh":
        d = ps_quote(suffix_dir)
        return [
            f"if (Test-Path {d}) {{ "
            f"if ($env:{var_name}) {{ $env:{var_name} = $env:{var_name} + {ps_quote(sep + suffix_dir)} }} "
            f"else {{ $env:{var_name} = {d} }} "
            f"}}"
        ]
    if shell == "cmd":
        d = suffix_dir
        return [
            f'if exist "{d}" (if defined {var_name} (set "{var_name}=%{var_name}%{sep}{d}") else (set "{var_name}={d}"))'
        ]
    word = f"${var_name}{sep}"
    return [
        f"if [ -d {sh_quote(suffix_dir)} ]; then export {var_name}=${{{var_name}:+{word}}}{sh_quote(suffix_dir)}; fi"
    ]


# --------------------------------------------------------------------------------------
# Subprocess helpers
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
    extra_env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> Optional[str]:
    run_env: Optional[dict[str, str]] = None
    if env is not None:
        run_env = dict(env)
    elif extra_env:
        run_env = os.environ.copy()

    if extra_env:
        assert run_env is not None
        run_env.update(extra_env)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=run_env,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True if capture else False,
    )
    if proc.returncode != 0:
        out = ""
        if capture and proc.stdout:
            out = str(proc.stdout)
        msg = f"Command failed ({proc.returncode}): {_format_cmd(cmd)}"
        if out:
            msg += "\n" + out
        raise CommandError(msg)

    return str(proc.stdout) if capture else None


def capture_text(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> str:
    return run_checked(cmd, cwd=cwd, env=env, capture=True) or ""


# --------------------------------------------------------------------------------------
# Venv + pip helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VenvInfo:
    venv_dir: Path
    python: Path


def venv_python_path(venv_dir: Path) -> Path:
    # Python layout differs between POSIX and Windows.
    return (venv_dir / "Scripts" / "python.exe") if IS_WINDOWS else (venv_dir / "bin" / "python")


def ensure_venv(venv_dir: Path, *, python_exe: str) -> VenvInfo:
    if not venv_dir.exists():
        run_checked([python_exe, "-m", "venv", str(venv_dir)])
    py = venv_python_path(venv_dir)
    if not py.exists():
        raise RuntimeError(f"venv python not found: {py}")
    return VenvInfo(venv_dir=venv_dir, python=py)


def pip_install(venv: VenvInfo, args: list[str]) -> None:
    run_checked([str(venv.python), "-m", "pip"] + args)


def pip_install_requirements(venv: VenvInfo, requirements: Path, *, upgrade: bool, force_reinstall: bool) -> None:
    cmd = ["install"]
    if upgrade:
        cmd.append("--upgrade")
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd += ["-r", str(requirements)]
    pip_install(venv, cmd)


def pip_install_package(
    venv: VenvInfo,
    package: str,
    *,
    upgrade: bool,
    force_reinstall: bool,
    find_links: Optional[str] = None,
    wheelhouse: Optional[Path] = None,
    no_index: bool = False,
    no_deps: bool = False,
) -> None:
    cmd = ["install"]
    if upgrade:
        cmd.append("--upgrade")
    if force_reinstall:
        cmd.append("--force-reinstall")
    if no_deps:
        cmd.append("--no-deps")
    if no_index:
        cmd.append("--no-index")
    if wheelhouse is not None:
        cmd += ["--find-links", str(wheelhouse)]
    if find_links:
        cmd += ["-f", find_links]
    cmd.append(package)
    pip_install(venv, cmd)


def pip_install_prefix(
    venv: VenvInfo,
    dist_name: str,
    candidates: list[str],
    *,
    require_subdir: Optional[str] = None,
) -> Optional[Path]:
    # Resolve a wheel install prefix from site-packages using `pip show`.
    try:
        out = capture_text([str(venv.python), "-m", "pip", "show", dist_name]).replace("\r", "")
    except CommandError:
        return None

    location = next(
        (line.split(":", 1)[1].strip() for line in out.splitlines() if line.lower().startswith("location:")),
        "",
    )
    if not location:
        return None

    base = Path(location).resolve()

    def _ok(prefix: Path) -> bool:
        return prefix.is_dir() and ((require_subdir is None) or (prefix / require_subdir).is_dir())

    for cand in candidates:
        cand = str(cand).strip()
        if not cand:
            continue
        p = (base / cand).resolve()
        if _ok(p):
            return p

    return None


def ensure_mlir_aie_pth(venv: VenvInfo, mlir_aie_install_dir: Path) -> None:
    python_dir = (mlir_aie_install_dir / "python").resolve()
    if not python_dir.is_dir():
        return
    try:
        site_pkgs = capture_text(
            [str(venv.python), "-c", "import sysconfig; p=sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib') or ''; print(p)"]
        ).strip()
        if not site_pkgs:
            return
        pth_path = Path(site_pkgs) / "mlir-aie.pth"
        desired = str(python_dir)
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        pth_path.write_text(desired + "\n", encoding="utf-8")
    except Exception:
        return


# --------------------------------------------------------------------------------------
# Windows patches
# --------------------------------------------------------------------------------------


def fixup_llvm_aie_windows(peano_root: Optional[Path]) -> None:
    # Windows-only llvm-aie wheel patches:
    #   - Create libc.a/libm.a aliases for c.lib/m.lib
    #   - Strip '.deplibs' from crt1.o if present (prevents bogus MSVC runtime deps)
    if not IS_WINDOWS or peano_root is None:
        return

    libroot = peano_root / "lib"
    if not libroot.is_dir():
        return
    # llvm-aie installs are organized by target triple under <prefix>/lib.
    toolchains = [p for p in libroot.iterdir() if p.is_dir() and p.name.endswith("-none-unknown-elf")]
    if not toolchains:
        print(f"[fixup] NOTE: no *-none-unknown-elf toolchains found under: {libroot} (skipping llvm-aie patches)")

    objcopy = peano_root / "bin" / "llvm-objcopy.exe"
    objcopy_exe = str(objcopy) if objcopy.exists() else (shutil.which("llvm-objcopy") or shutil.which("llvm-objcopy.exe"))
    if not objcopy_exe:
        print(f"[fixup] NOTE: llvm-objcopy not found; skipping crt1.o patch.")

    for libdir in toolchains:
        if not libdir.is_dir():
            continue
        for src_name, dst_name in (("c.lib", "libc.a"), ("m.lib", "libm.a")):
            src = libdir / src_name
            dst = libdir / dst_name
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    print(f"[fixup] Created alias in {libdir.name}: {dst.name} (copy of {src.name})")
                except Exception as e:
                    print(f"[fixup] WARNING: failed to create {dst} from {src}: {e}")

        crt1 = libdir / "crt1.o"
        if not crt1.exists() or not objcopy_exe:
            continue
        bak = crt1.with_suffix(crt1.suffix + ".bak")
        if not bak.exists():
            try:
                shutil.copy2(crt1, bak)
            except Exception:
                pass

        proc = subprocess.run(
            [objcopy_exe, "--remove-section=.deplibs", str(crt1)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            print(f"[fixup] Patched {libdir.name}/crt1.o: removed '.deplibs' (if present)")


# --------------------------------------------------------------------------------------
# Install/update
# --------------------------------------------------------------------------------------


# `--all` enables everything, but `--no-<x>` explicitly opts out of sub-features.
def _apply_all_flag(args: argparse.Namespace, argv: list[str]) -> None:
    if not getattr(args, "all", False):
        return
    for flag in ("repo-reqs", "dev", "ml", "notebook", "submodules"):
        dest = flag.replace("-", "_")
        if (f"--{flag}" in argv) or (f"--no-{flag}" in argv):
            continue
        setattr(args, dest, True)


def update_submodules(repo_root: Path) -> None:
    if (repo_root / ".git").exists() and shutil.which("git"):
        print("[setup] Updating git submodules...")
        run_checked(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_root)
    else:
        print("[setup] NOTE: Skipping submodules (not a git checkout)")


def print_next_steps(repo_root: Path, *, venv_name: str) -> None:
    script_path = Path(__file__).resolve()
    try:
        script_rel = script_path.relative_to(repo_root)
    except Exception:
        script_rel = script_path
    script_str = str(script_rel)

    print("\n[setup] Setup complete. Set environment variables for your current shell session with:")
    print(f"  # PowerShell:  python {script_str} env --venv {venv_name} --shell pwsh | iex")
    print(fr'  # cmd.exe   :  python {script_str} env --venv {venv_name} --shell cmd > "%TEMP%\iron_env.bat" && call "%TEMP%\iron_env.bat"')
    print(f'  # POSIX sh  :  eval "$(python3 {script_str} env --venv {venv_name} --shell sh)"')
    print("  #              source /opt/xilinx/xrt/setup.sh")

    print("\n[setup] To update later:")
    print(f"  python {script_str} update")


def install_plan(args: argparse.Namespace, repo_root: Path, *, update_mode: bool) -> None:
    _apply_all_flag(args, sys.argv[1:])

    venv_dir = (repo_root / args.venv).resolve()
    venv = ensure_venv(venv_dir, python_exe=args.python)

    print(f"[setup] repo_root : {repo_root}")
    print(f"[setup] venv      : {venv.venv_dir}")
    print(f"[setup] wsl       : {IS_WSL}")
    print(f"[setup] mode      : {'update' if update_mode else 'install'}")

    force_reinstall = bool(getattr(args, "force_reinstall", False))
    pip_reqs = lambda req: pip_install_requirements(venv, req, upgrade=update_mode, force_reinstall=force_reinstall)
    pip_pkg = lambda pkg, **kw: pip_install_package(venv, pkg, upgrade=update_mode, force_reinstall=force_reinstall, **kw)

    # Base tooling.
    pip_install(venv, ["install", "--upgrade", "pip", "setuptools", "wheel", "packaging"])

    # Repo requirements + optional extras.
    req_specs = [
        ("repo_reqs", "python/requirements.txt", None, None),
        ("dev", "python/requirements_dev.txt", ["pre_commit", "install"], repo_root),
        ("ml", "python/requirements_ml.txt", None, None),
        ("notebook", "python/requirements_notebook.txt", ["ipykernel", "install", "--user", "--name", args.venv], None),
    ]

    for flag, req_rel, post_mod_args, post_cwd in req_specs:
        if not getattr(args, flag, False):
            continue
        req = (repo_root / req_rel).resolve()
        if not req.exists():
            print(f"[setup] NOTE: Skipping {flag}; missing: {req}")
            continue

        pip_reqs(req)

        if post_mod_args:
            try:
                run_checked([str(venv.python), "-m", *post_mod_args], cwd=post_cwd)
            except Exception:
                print(f"[setup] WARNING: post-step for {flag} failed (continuing)")

    # mlir_aie wheels (IRON)
    # Modes: auto | skip | latest-wheels-3 | wheelhouse[:<path>]
    # Auto mode exists until wheels are available for Windows.
    mlir_raw = str(getattr(args, "mlir_aie", "auto") or "auto").strip().lower()
    mlir_mode, allow_missing_mlir_aie = (("latest-wheels-3", IS_WINDOWS) if mlir_raw == "auto" else (mlir_raw, False))
    if mlir_mode != "skip":
        try:
            if mlir_mode == "wheelhouse" or mlir_mode.startswith("wheelhouse:"):
                wheelhouse_dir = Path(mlir_mode.partition(":")[2] or (repo_root / "utils" / "mlir_aie_wheels" / "wheelhouse")).expanduser()
                if not wheelhouse_dir.exists(): raise RuntimeError(f"Wheelhouse directory not found: {wheelhouse_dir}")
                print(f"[setup] Installing mlir_aie + aie_python_bindings from wheelhouse: {wheelhouse_dir}")
                for pkg in ("mlir_aie", "aie_python_bindings"):
                    pip_pkg(pkg, wheelhouse=wheelhouse_dir, no_deps=True, no_index=True)
            else:
                mlir_find_links = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3/"
                print(f"[setup] Installing mlir_aie from {mlir_find_links}")
                pip_pkg("mlir_aie", find_links=mlir_find_links)
        except CommandError:
            if allow_missing_mlir_aie:
                print("[setup] NOTE: mlir_aie wheels not available for this platform/Python (continuing).")
            else:
                raise
        if (mlir_prefix := pip_install_prefix(venv, "mlir_aie", ["mlir_aie"])):
            ensure_mlir_aie_pth(venv, mlir_prefix)

    # llvm-aie wheels (Peano)
    llvm_choice = str(getattr(args, "llvm_aie", "nightly") or "nightly").strip()
    llvm_find_links = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly" if llvm_choice == "nightly" else llvm_choice

    print(f"[setup] Installing llvm-aie from {llvm_find_links}")
    pip_pkg("llvm-aie", find_links=llvm_find_links)

    if IS_WINDOWS:
        peano_prefix = pip_install_prefix(venv, "llvm-aie", ["llvm-aie", "llvm_aie"], require_subdir="bin")
        fixup_llvm_aie_windows(peano_prefix)

    if getattr(args, "submodules", False):
        update_submodules(repo_root)

    print_next_steps(repo_root, venv_name=args.venv)


# --------------------------------------------------------------------------------------
# NPU detection
# --------------------------------------------------------------------------------------


# If we have anything that reports like an NPU, even if unknown type, accept it as at least XDNA.
NPU_REGEX = re.compile(r"(RyzenAI-npu(?![0-3]\b)\d+|\bNPU\b)", re.IGNORECASE)
NPU2_REGEX = re.compile(r"NPU Strix|NPU Strix Halo|NPU Krackan|NPU Gorgon|RyzenAI-npu[4567]", re.IGNORECASE)
NPU3_REGEX = re.compile(r"NPU Medusa|RyzenAI-npu[8]", re.IGNORECASE)


# Windows and WSL must use a driver-provided xrt-smi.exe in System32\AMD.
def system32_amd_xrt_smi_dir() -> Optional[Path]:
    if IS_WINDOWS:
        p = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
        return p.parent if p.exists() else None
    if IS_WSL:
        p = Path("/mnt/c/Windows/System32/AMD/xrt-smi.exe")
        return p.parent if p.exists() else None
    return None


def xrt_smi_commands() -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def _add(cmd: list[str]) -> None:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            out.append(cmd)

    for exe in ("xrt-smi", "xrt-smi.exe"):
        if shutil.which(exe):
            _add([exe, "examine"])

    if (sys32 := system32_amd_xrt_smi_dir()):
        _add([str(sys32 / "xrt-smi.exe"), "examine"])

    return out


# Detect NPU2 (or greater) capability via xrt-smi output.
def detect_npu2_flag() -> tuple[Optional[bool], str]:
    for cmd in xrt_smi_commands():
        try:
            out = capture_text(cmd).replace("\r", "")
        except Exception:
            continue

        # For now, we detect but do not support NPU3.
        if NPU3_REGEX.search(out):
            return False, f"Detected unsupported device via: {_format_cmd(cmd)}"

        if NPU_REGEX.search(out):
            if NPU2_REGEX.search(out):
                return True, f"Detected NPU2 device via: {_format_cmd(cmd)}"
            return False, f"Detected NPU device via: {_format_cmd(cmd)}"

    return None, "WARNING: xrt-smi not available (or no NPU detected)"


# --------------------------------------------------------------------------------------
# Emit env for current shell session
# --------------------------------------------------------------------------------------


def _resolve_windows_xrt_root(args: argparse.Namespace) -> Optional[Path]:
    # Prefer --xrt-root / XRT_ROOT. Only export it if the directory exists.
    # NOTE: NEVER set XILINX_XRT on Windows.
    root = (getattr(args, "xrt_root", "") or os.environ.get("XRT_ROOT") or "").strip()
    if root:
        p = Path(root).expanduser()
        p = p.resolve() if p.is_absolute() else p
        return p if p.exists() else None

    default = Path(r"C:/Xilinx/XRT")
    return default if default.exists() else None


def env_plan(args: argparse.Namespace, repo_root: Path) -> None:
    venv_dir = (repo_root / args.venv).resolve()
    venv_py = venv_python_path(venv_dir)

    shell = (getattr(args, "shell", "auto") or "auto").lower()
    if shell == "auto":
        shell = default_env_shell()

    comment = "REM" if shell == "cmd" else "#"
    out_lines: list[str] = []

    # Activate venv.
    if venv_py.exists():
        out_lines.append(f"{comment} Activate venv: {venv_dir}")
        if shell == "pwsh":
            activate = venv_dir / "Scripts" / "Activate.ps1"
            out_lines.append(f"if (Test-Path {ps_quote(str(activate))}) {{ . {ps_quote(str(activate))} }}")
        elif shell == "cmd":
            activate = venv_dir / "Scripts" / "activate.bat"
            q = cmd_quote(str(activate))
            out_lines.append(f"if exist {q} call {q}")
        else:
            activate = (venv_dir / "bin" / "activate")
            out_lines.append(f"if [ -f {sh_quote(str(activate))} ]; then source {sh_quote(str(activate))}; fi")
    else:
        out_lines.append(f"{comment} WARNING: venv not found at: {venv_dir} (run: `python utils/iron_setup.py install`)")

    mlir_prefix: Optional[Path] = None
    peano_prefix: Optional[Path] = None

    if venv_py.exists():
        venv = VenvInfo(venv_dir=venv_dir, python=venv_py)
        mlir_prefix = pip_install_prefix(venv, "mlir_aie", ["mlir_aie"])
        peano_prefix = pip_install_prefix(venv, "llvm-aie", ["llvm-aie", "llvm_aie"], require_subdir="bin")

    def _override(raw: str, valid, note: str) -> Optional[Path]:
        raw = (raw or "").strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.is_dir() and valid(p):
            return p
        out_lines.append(f"{comment} NOTE: {note}: {p}")
        return None

    # Optional explicit overrides.
    override = _override(getattr(args, "mlir_aie_install", ""), lambda p: (p / "bin").is_dir() or (p / "python").is_dir(), "Ignoring --mlir-aie-install (not a valid prefix)")
    if override:
        mlir_prefix = override

    override = _override(getattr(args, "llvm_aie_install", ""), lambda p: (p / "bin").is_dir(), "Ignoring --llvm-aie-install (missing bin/)")
    if override:
        peano_prefix = override

    # Driver-provided tools on PATH (kept first).
    driver_dir = system32_amd_xrt_smi_dir()
    if driver_dir:
        out_lines.extend(emit_prepend_path(shell, "PATH", str(driver_dir)))

    # mlir-aie env.
    if mlir_prefix:
        out_lines.extend(emit_set(shell, "MLIR_AIE_INSTALL_DIR", str(mlir_prefix)))
        out_lines.extend(emit_prepend_path(shell, "PATH", str(mlir_prefix / "bin")))
        out_lines.extend(emit_prepend_path(shell, "PYTHONPATH", str(mlir_prefix / "python")))
        lib_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
        out_lines.extend(emit_prepend_path(shell, lib_var, str(mlir_prefix / "lib")))
    else:
        out_lines.append(f"{comment} NOTE: mlir_aie not installed in this venv (or not detected).")

    # llvm-aie env.
    if peano_prefix:
        out_lines.extend(emit_set(shell, "PEANO_INSTALL_DIR", str(peano_prefix)))
        out_lines.append(f"{comment} NOTE: llvm-aie is not added to PATH to avoid conflicts with system clang/clang++.")
        out_lines.append(f"{comment}       It can be found in: {str(peano_prefix / 'bin')}")
    else:
        out_lines.append(f"{comment} WARNING: llvm-aie not installed in this venv (run: `python utils/iron_setup.py install`)")

    # XRT env.
    if IS_WINDOWS:
        # XILINX_XRT must not be set on Windows.
        if shell == "pwsh":
            out_lines.append(r"Remove-Item Env:\XILINX_XRT -ErrorAction SilentlyContinue")
        elif shell == "cmd":
            out_lines.append('set "XILINX_XRT="')
        else:
            out_lines.append("unset XILINX_XRT 2>/dev/null || true")

        xrt_root = _resolve_windows_xrt_root(args)
        if xrt_root:
            out_lines.extend(emit_set(shell, "XRT_ROOT", str(xrt_root)))
            for p in (xrt_root / "ext" / "bin", xrt_root / "lib", xrt_root / "unwrapped", xrt_root):
                out_lines.extend(emit_append_path_if_exists(shell, "PATH", str(p)))
            out_lines.extend(emit_append_path_if_exists(shell, "PYTHONPATH", str(xrt_root / "python")))
    else:
        xrt_root = (getattr(args, "xrt_root", "") or os.environ.get("XRT_ROOT") or os.environ.get("XILINX_XRT") or "").strip()
        if xrt_root:
            out_lines.extend(emit_set(shell, "XRT_ROOT", xrt_root))
            out_lines.extend(emit_set(shell, "XILINX_XRT", xrt_root))
        if Path("/opt/xilinx/xrt/setup.sh").exists():
            out_lines.append(f"{comment} NOTE: If XRT tools/DLLs are not visible, run:")
            out_lines.append(f"{comment}   source /opt/xilinx/xrt/setup.sh")

    # NPU detection.
    npu2_flag, reason = detect_npu2_flag()
    if npu2_flag is not None:
        out_lines.extend(emit_set(shell, "NPU2", "1" if npu2_flag else "0"))
    out_lines.append(f"{comment} NOTE: {reason}")

    print("\n".join(out_lines))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_bool_flag_pair(
    p: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_on: str,
    help_off: str,
) -> None:
    dest = name.replace("-", "_")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=dest, action="store_true", help=help_on)
    grp.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_off)
    p.set_defaults(**{dest: default})


def _add_install_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--python", default=sys.executable, help="Python executable used to create the venv")
    p.add_argument("--venv", default="ironenv", help="Venv directory name relative to repo root")
    p.add_argument("--mlir-aie", default="auto", help="mlir_aie wheel source: auto | skip | latest-wheels-3 | wheelhouse[:<path>]")
    p.add_argument("--llvm-aie", default="nightly", help="llvm-aie wheels: nightly (default) or a custom -f URL")
    p.add_argument("--force-reinstall", dest="force_reinstall", action="store_true", help="Force reinstall packages (pip --force-reinstall). Useful for testers.",)
    p.add_argument("--all", action="store_true", help="Enable everything (repo-reqs + dev + ml + notebook + submodules).")

    _add_bool_flag_pair(p, "repo-reqs", default=True, help_on="Install python/requirements.txt (repo Python deps).", help_off="Skip repo Python deps install.")
    _add_bool_flag_pair(p, "dev", default=True, help_on="Install python/requirements_dev.txt + pre-commit.", help_off="Skip dev deps.")
    _add_bool_flag_pair(p, "ml", default=False, help_on="Install python/requirements_ml.txt.", help_off="Skip ML deps.")
    _add_bool_flag_pair(p, "notebook", default=False, help_on="Install python/requirements_notebook.txt + (best-effort) ipykernel registration.", help_off="Skip notebook deps.",)
    _add_bool_flag_pair(p, "submodules", default=False, help_on="Update git submodules if this is a git checkout.", help_off="Do not update git submodules.",)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="cmd")

    p_install_common = argparse.ArgumentParser(add_help=False)
    _add_install_args(p_install_common)

    p_install = sub.add_parser("install", parents=[p_install_common], help="Install toolchain wheels + deps into the venv")
    p_update = sub.add_parser("update", parents=[p_install_common], help="Update to newest wheels + deps in the existing venv")
    p_update.set_defaults(submodules=True)

    p_env = sub.add_parser("env", help="Print shell commands to activate venv + export toolchain env vars")
    p_env.add_argument("--venv", default="ironenv", help="Venv directory relative to repo root")
    p_env.add_argument("--xrt-root", default="", help="Optional XRT install directory (Windows: default C:/Xilinx/XRT). Can also be set via XRT_ROOT.",)
    p_env.add_argument("--mlir-aie-install", default="", help="Optional explicit mlir-aie install prefix (relative to repo root unless absolute).")
    p_env.add_argument("--llvm-aie-install", default="", help="Optional explicit llvm-aie/peano install prefix (relative to repo root unless absolute).")
    p_env.add_argument("--shell", default="auto", choices=["auto", "sh", "pwsh", "cmd"], help="Which shell syntax to emit (default: auto based on platform)")

    return p


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = build_arg_parser()

    # Allow calling without an explicit subcommand (defaults to install).
    argv = sys.argv[1:]
    subcommands = {"install", "update", "env"}

    if not argv:
        argv = ["install"]
    elif argv[0] not in subcommands:
        # If the user is asking for top-level help, don't force the install help.
        if any(a in ("-h", "--help") for a in argv):
            parser.print_help()
            return 0
        argv = ["install"] + argv

    args = parser.parse_args(argv)
    if IS_WINDOWS:
        # XILINX_XRT poisons Windows builds.
        os.environ.pop("XILINX_XRT", None)

    if args.cmd is None or args.cmd == "install":
        install_plan(args, repo_root, update_mode=False)
        return 0
    if args.cmd == "update":
        install_plan(args, repo_root, update_mode=True)
        return 0
    if args.cmd == "env":
        env_plan(args, repo_root)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
