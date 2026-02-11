#!/usr/bin/env python3
##===------ run_example.py - Runs IRON examples (Linux/WSL/Windows). -----===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===------------------------------------------------------------------------------===##
#
# Build + run Makefile-based programming examples (cross platform).
#
# This script is not a Makefile interpreter. It uses the example Makefile as a hint to
# discover the target name, VPATH search roots, and kernel object prerequisites.
#
# It attempts to perform the same basic flow as MOST examples:
#   1) Generate AIE MLIR by running the example's Python generator.
#   2) Compile kernel objects (e.g. scale.cc -> build/scale.o).
#   3) Invoke aiecc to produce:
#        - build/final_<data_size>.xclbin
#        - build/insts_<data_size>.bin
#   4) Optionally build a host executable via CMake and run it.
#
# Usage
# -----
# From an example directory containing a Makefile:
#
#   # PowerShell
#   python path\to\run_example.py --example-dir . build
#   python path\to\run_example.py --example-dir . run
#
# Or default to the current directory:
#   python path\to\run_example.py build
#
##===------------------------------------------------------------------------------===##


from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------------------
# Platform + process helpers
# --------------------------------------------------------------------------------------


def is_windows() -> bool:
    return os.name == "nt"


def is_wsl() -> bool:
    if is_windows():
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        osrelease = Path("/proc/sys/kernel/osrelease").read_text(errors="ignore").lower()
        return "microsoft" in osrelease or "wsl" in osrelease
    except Exception:
        return False


def _wsl_to_windows_path(path: Path) -> str:
    # wslpath is present inside WSL; avoid depending on it elsewhere.
    if not is_wsl():
        return str(path)
    wslpath_exe = which("wslpath")
    if not wslpath_exe:
        raise RuntimeError("wslpath not found; cannot translate WSL paths for Windows tools")
    out = subprocess.check_output([wslpath_exe, "-w", str(path)], text=True).strip()
    if not out:
        raise RuntimeError(f"wslpath returned empty path for: {path}")
    return out


def which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def format_cmd(cmd: list[str]) -> str:
    # For logging only; quoting does not need to be shell-perfect.
    try:
        return shlex.join(cmd)
    except AttributeError:
        return " ".join(shlex.quote(c) for c in cmd)


def tool_exe_name(base: str) -> str:
    # Tools need `.exe` appended on Windows
    return base + (".exe" if is_windows() else "")


def host_exe_name(base: str) -> str:
    # Host needs `.exe` appended on Windows
    return base + (".exe" if (is_windows() or is_wsl()) else "")


def _cmake_generator_from_cache(build_dir: Path) -> Optional[str]:
    # Read CMAKE_GENERATOR from an existing CMakeCache.txt (if present).
    cache = build_dir / "CMakeCache.txt"
    try:
        for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                return line.split("=", 1)[1].strip() or None
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def _is_multi_config_generator(generator: str) -> bool:
    g = (generator or "").lower()
    return ("visual studio" in g) or ("xcode" in g) or ("ninja multi-config" in g)



def run_checked(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> None:
    print(f"[run] cwd={cwd or Path.cwd()}")
    print(f"[run] $ {format_cmd(cmd)}")
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _max_mtime(paths: list[Path]) -> float:
    mt = 0.0
    for p in paths:
        try:
            mt = max(mt, p.stat().st_mtime)
        except FileNotFoundError:
            pass
    return mt


def _outputs_up_to_date(outputs: list[Path], inputs: list[Path]) -> bool:
    if not outputs:
        return False
    for o in outputs:
        if not o.exists():
            return False
    in_mt = _max_mtime(inputs)
    out_mt = min(o.stat().st_mtime for o in outputs)
    return out_mt >= in_mt


def _up_to_date_with_stamp(
    outputs: list[Path],
    inputs: list[Path],
    stamp_path: Optional[Path] = None,
    stamp_data: Optional[dict] = None,
) -> bool:
    # Return True when outputs are newer than inputs and (optionally) the stamp matches.
    if not _outputs_up_to_date(outputs, inputs):
        return False
    if stamp_path is not None and stamp_data is not None:
        return _stamp_matches(stamp_path, stamp_data)
    return True


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_json_atomic(path: Path, data: dict) -> None:
    # Keep stamps readable for debugging; write atomically to avoid partial files.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _stamp_matches(stamp_path: Path, data: dict) -> bool:
    old = _read_json(stamp_path)
    return old == data


def _looks_like_flag_incompatible(stderr_text: str) -> bool:
    # Heuristics for legacy generators that do NOT use argparse flags.
    # Keep this broad: if we miss the retry, we break older examples.
    t = (stderr_text or "").lower()
    if "unrecognized arguments" in t:
        return True
    if "no such option" in t:
        return True
    # Some scripts treat "-d" as the device name and reject it as unknown.
    if "device name" in t and "unknown" in t and "-d" in t:
        return True
    if "usage:" in t and ("-d" in t or "-i1s" in t or "-bw" in t):
        return True
    if "need " in t and " command line argument" in t:
        return True
    if "indexerror" in t and "list index out of range" in t:
        return True
    return False


def _run_to_file(cmd: list[str], cwd: Path, out_path: Path) -> subprocess.CompletedProcess[str]:
    # Write to a temp file and replace on success so failed attempts don't clobber good artifacts.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
        )
    if proc.returncode == 0:
        os.replace(tmp, out_path)
    else:
        try:
            tmp.unlink()
        except Exception:
            pass
    return proc


# --------------------------------------------------------------------------------------
# Makefile discovery (pattern-based)
# --------------------------------------------------------------------------------------


@dataclass
class ExampleMakeInfo:
    example_dir: Path
    targetname: str
    aie_py_src: Optional[str]
    vpath_dirs: list[Path]
    kernel_object_names: list[str]
    has_trace: bool
    # Runlist-style examples often generate an ELF (insts.elf) instead of NPU insts (.bin).
    uses_elf_insts: bool
    xclbin_name_hint: Optional[str]
    insts_name_hint: Optional[str]

    # Makefile defaults (best-effort, optional)
    default_int_bit_width: Optional[int]
    default_in2_size: Optional[int]
    default_trace_size: Optional[int]
    default_col: Optional[int]

_TARGET_RE = re.compile(r"^\s*targetname\s*[:?]?=\s*(\S+)\s*$")
_AIE_PY_SRC_RE = re.compile(r"^\s*aie_py_src\s*[:?]?=\s*(\S+)\s*$")
_FINAL_RULE_RE = re.compile(r"^\s*(build/final[^:]*\.xclbin)\s*:\s*(.+?)\s*$")
_INT_BW_RE = re.compile(r"^\s*int_bit_width\s*[:?]?=\s*(\d+)\s*(?:#.*)?$")
_IN2_RE = re.compile(r"^\s*in2_size\s*[:?]?=\s*(\d+)\s*(?:#.*)?$")
_TRACE_RE = re.compile(r"^\s*trace_size\s*[:?]?=\s*(\d+)\s*(?:#.*)?$")
_COL_RE = re.compile(r"^\s*col\s*[:?]?=\s*(\d+)\s*(?:#.*)?$", re.IGNORECASE)
_VAR_REF_RE = re.compile(r"\$\(([^)]+)\)|\${([^}]+)}")
_SIMPLE_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\?|\+|:)?=\s*(.+?)\s*$")


def expand_make_vars(value: str, vars_map: dict[str, str]) -> str:
    # ${var}/$(var) expansion.
    # This is NOT a full Make interpreter; it only helps with common patterns like:
    #   targetname=${orig_targetname}_chained
    cur = value
    for _ in range(6):
        def _repl(m: re.Match) -> str:
            name = m.group(1) or m.group(2) or ""
            return vars_map.get(name, m.group(0))

        nxt = _VAR_REF_RE.sub(_repl, cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


def _collapse_makefile_lines(lines: list[str]) -> list[str]:
    # Merge line continuations ending in '\\'.
    out: list[str] = []
    cur = ""
    for raw in lines:
        line = raw.rstrip("\n")
        if cur:
            line = cur + line.lstrip()
        if line.rstrip().endswith("\\"):
            cur = line.rstrip()[:-1] + " "
            continue
        cur = ""
        out.append(line)
    if cur:
        out.append(cur)
    return out


def _parse_vpath(line: str, example_dir: Path, vars_map: dict[str, str]) -> list[Path]:
    # Accept: VPATH =, VPATH :=, VPATH ?=, VPATH +=
    m = re.match(r"^\s*VPATH\s*(?:\?|\+|:)?=\s*(.+?)\s*$", line)
    if not m:
        return []

    vpath_raw = m.group(1).strip()
    if not vpath_raw:
        return []

    # Make VPATH may be a single path or colon-separated paths.
    vpath_raw = expand_make_vars(vpath_raw, vars_map)
    raw_parts = [p.strip() for p in vpath_raw.split(":") if p.strip()]
    vpath_dirs: list[Path] = []
    for p in raw_parts:
        # Expand ${srcdir}/$(srcdir) even if it wasn't in the vars map.
        p = p.replace("${srcdir}", str(example_dir)).replace("$(srcdir)", str(example_dir))
        vpath_dirs.append(Path(p).resolve())
    return vpath_dirs


def parse_makefile(example_dir: Path) -> ExampleMakeInfo:
    mk = example_dir / "Makefile"
    if not mk.exists():
        raise FileNotFoundError(f"Makefile not found in: {example_dir}")

    txt = read_text(mk)
    lines = _collapse_makefile_lines(txt.splitlines())

    vars_map: dict[str, str] = {
        # Many Makefiles reference ${srcdir}/$(srcdir). We treat it as the
        # example directory (matching upstream Makefile conventions).
        "srcdir": str(example_dir.resolve()),
    }

    targetname: Optional[str] = None
    aie_py_src: Optional[str] = None
    vpath_dirs: list[Path] = []
    kernel_objs: list[str] = []
    has_trace = False
    # Runlist-style examples use --aie-generate-elf and pass an ELF to the host (-i insts.elf).
    uses_elf_insts = "--aie-generate-elf" in txt
    xclbin_name_hint: Optional[str] = None
    insts_name_hint: Optional[str] = None


    def _sanitize_make_token(tok: str) -> Optional[str]:
        # Ignore Make variables like ${@F} / ${<F}.
        if not tok:
            return None
        if any(ch in tok for ch in ("$", "{", "}", "(", ")")):
            return None
        return tok

    # Best-effort hints for output file names (used mainly for runlist/ELF examples).
    m = re.search(r"-x\s+([^\s\\]+\.xclbin)", txt)
    if m:
        xclbin_name_hint = _sanitize_make_token(Path(m.group(1)).name)

    m = re.search(r"-i\s+([^\s\\]+\.(?:elf|bin))", txt)
    if m:
        insts_name_hint = _sanitize_make_token(Path(m.group(1)).name)

    # If the Makefile explicitly names the ELF, prefer that.
    m = re.search(r"--elf-name(?:=|\s+)([^\s\\]+)", txt)
    if m:
        insts_name_hint = _sanitize_make_token(Path(m.group(1)).name) or insts_name_hint

    default_bw: Optional[int] = None
    default_in2: Optional[int] = None
    default_trace: Optional[int] = None
    default_col: Optional[int] = None

    for line in lines:
        # Capture simple (non-$) assignments to support basic ${var} expansions.
        m_assign = _SIMPLE_ASSIGN_RE.match(line)
        if m_assign:
            name = m_assign.group(1)
            value = m_assign.group(2).strip()
            if name not in vars_map and "$" not in value:
                vars_map[name] = value

        if targetname is None:
            m = _TARGET_RE.match(line)
            if m:
                targetname = expand_make_vars(m.group(1).strip(), vars_map)
                continue

        if aie_py_src is None:
            m = _AIE_PY_SRC_RE.match(line)
            if m:
                aie_py_src = expand_make_vars(m.group(1).strip(), vars_map)
                continue

        # Keep the first VPATH definition we find (most examples only define one).
        if not vpath_dirs:
            vpath_dirs = _parse_vpath(line, example_dir, vars_map)
            if vpath_dirs:
                continue

        m = _INT_BW_RE.match(line)
        if m and default_bw is None:
            default_bw = int(m.group(1))
            continue

        m = _IN2_RE.match(line)
        if m and default_in2 is None:
            default_in2 = int(m.group(1))
            continue

        m = _TRACE_RE.match(line)
        if m and default_trace is None:
            default_trace = int(m.group(1))
            continue

        m = _COL_RE.match(line)
        if m and default_col is None:
            default_col = int(m.group(1))
            continue

        # Discover kernel object prerequisites from the "final" rule.
        # Example:
        #   build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/scale.o
        m = _FINAL_RULE_RE.match(line)
        if m:
            rule_target = m.group(1).strip()
            deps = m.group(2).strip().split()
            # If the output name wasn't discovered from the run recipe, use the rule target.
            if not xclbin_name_hint:
                xclbin_name_hint = _sanitize_make_token(Path(rule_target).name)

            for dep in deps:
                dep = dep.strip()
                if dep.endswith(".o"):
                    name = Path(dep).stem
                    if name not in kernel_objs:
                        kernel_objs.append(name)

        if "final_trace" in line and ".xclbin" in line and ":" in line:
            has_trace = True

    if not targetname:
        # Not all examples define 'targetname ='. Fall back to the directory name.
        # If that guess is wrong, the user can still override via --targetname.
        targetname = example_dir.name

    return ExampleMakeInfo(
        example_dir=example_dir.resolve(),
        targetname=targetname,
        aie_py_src=aie_py_src,
        vpath_dirs=vpath_dirs,
        kernel_object_names=kernel_objs,
        has_trace=has_trace,
        uses_elf_insts=uses_elf_insts,
        xclbin_name_hint=xclbin_name_hint,
        insts_name_hint=insts_name_hint,
        default_int_bit_width=default_bw,
        default_in2_size=default_in2,
        default_trace_size=default_trace,
        default_col=default_col,
    )


# --------------------------------------------------------------------------------------
# Toolchain discovery
# --------------------------------------------------------------------------------------


def _env_activation_hint() -> str:
    if is_windows():
        return (
            "Activate the mlir-aie environment first (sets MLIR_AIE_INSTALL_DIR, PEANO_INSTALL_DIR, XRT_ROOT, PATH):\n"
            "  python utils/iron_setup.py env --shell pwsh | iex"
        )
    return (
        "Activate the mlir-aie environment first (sets MLIR_AIE_INSTALL_DIR, PEANO_INSTALL_DIR, XRT_ROOT, PATH):\n"
        "  eval \"$(python3 utils/iron_setup.py env --shell bash)\""
    )


# --------------------------------------------------------------------------------------
# Action groups
# --------------------------------------------------------------------------------------

TRACE_ACTIONS = {"trace", "trace_py"}
AIE_ACTIONS = {"build", "run", "run_py", *TRACE_ACTIONS}
HOST_BUILD_ACTIONS = {"run", "trace"}
RUN_HOST_ACTIONS = {"run", "trace"}
RUN_PY_ACTIONS = {"run_py", "trace_py"}


def preflight_or_die(action: str) -> None:
    # Only require tools that the requested action actually uses. This keeps
    # 'host' (cmake-only) usable even when XRT packaging tools aren't installed.
    need_aiecc = action in AIE_ACTIONS
    need_kernel_compile = action in AIE_ACTIONS

    missing: list[str] = []

    if need_aiecc:
        try:
            importlib.import_module("aie.compiler.aiecc.main")
        except Exception:
            missing.append("python package 'aie' (mlir-aie) not importable")

        if not which("bootgen"):
            missing.append("bootgen (required by aiecc to package .xclbin)")
        if not which("xclbinutil"):
            missing.append("xclbinutil (required by aiecc; usually from XRT/Vitis/Ryzen AI)")

    if need_kernel_compile:
        peano = resolve_peano_install_dir()
        try:
            _ = resolve_aie_clangpp(peano)
        except Exception:
            missing.append("clang++ (Peano/llvm-aie)")

    if missing:
        msg = (
            "Preflight failed; missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\n"
            + _env_activation_hint()
        )
        raise RuntimeError(msg)


def resolve_mlir_aie_root() -> Path:
    # Prefer the env var exported by `utils/iron_setup.py env`.
    env_root = (os.environ.get("MLIR_AIE_INSTALL_DIR") or "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if (p / "python").exists() or (p / "include").exists():
            return p

    # Equivalent to the Makefile's:
    #   python3 -c "from aie.utils.config import root_path; print(root_path())"
    try:
        from aie.utils.config import root_path  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "mlir-aie python package 'aie' not importable. Run iron_setup.py env (or install mlir-aie into this Python)."
        ) from e

    return Path(root_path()).resolve()


def resolve_peano_install_dir() -> Optional[Path]:
    # Prefer env vars exported by utils/iron_setup.py env.
    v = (os.environ.get("PEANO_INSTALL_DIR") or "").strip()
    if v:
        p = Path(v).resolve()
        if (p / "bin").exists():
            return p
    return None


def resolve_aie_clangpp(peano_install: Optional[Path]) -> str:
    if peano_install is not None:
        exe = peano_install / "bin" / tool_exe_name("clang++")
        if exe.exists():
            return str(exe)

    exe = which("clang++") or which("clang++.exe")
    if exe:
        return exe

    raise RuntimeError(
        "clang++ not found (llvm-aie/Peano). Run iron_setup.py env or put PEANO_INSTALL_DIR/bin on PATH."
    )


def xchesscc_available() -> bool:
    return bool(which("xchesscc") or which("xchesscc.exe"))


def build_aiecc_command(args_for_aiecc: list[str]) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from aie.compiler.aiecc.main import main; main()",
        *args_for_aiecc,
    ]


# --------------------------------------------------------------------------------------
# Build steps
# --------------------------------------------------------------------------------------


@dataclass
class BuildConfig:
    example_dir: Path
    targetname: str
    device: str
    col: int
    int_bit_width: int
    in1_size: int
    in2_size: int
    out_size: int
    trace_size: int
    placed: bool
    chess: bool
    config: str
    generator: Optional[str]
    generator_script: Optional[Path]
    extra_generator_args: list[str] = field(default_factory=list)


def default_sizes_for_bitwidth(int_bit_width: int) -> tuple[int, int, int]:
    # Mirrors common example defaults.
    if int_bit_width == 16:
        return (8192, 4, 8192)
    if int_bit_width == 32:
        return (16384, 4, 16384)
    # Fallback: keep the 16-bit layout.
    return (8192, 4, 8192)


def resolve_kernel_sources(info: ExampleMakeInfo) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for name in info.kernel_object_names:
        exts = [".cc", ".cpp", ".cxx", ".c"]
        candidates: list[Path] = []
        for ext in exts:
            fname = f"{name}{ext}"
            candidates.extend([info.example_dir / fname] + [vp / fname for vp in info.vpath_dirs])
        found: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                found = cand.resolve()
                break

        if not found:
            raise FileNotFoundError(
                f"Could not locate kernel source for '{name}.[cc/cpp/cxx/c]' in example dir or VPATH.\n"
                f"Searched: {candidates}"
            )

        sources.append((name, found))

    return sources


def compile_kernel_objects(cfg: BuildConfig, info: ExampleMakeInfo, build_dir: Path) -> None:
    mlir_aie_root = resolve_mlir_aie_root()
    peano_install = resolve_peano_install_dir()
    clangpp = resolve_aie_clangpp(peano_install)
    warning_flags = [
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
    ]
    if cfg.device == "npu2":
        target_flag = "--target=aie2p-none-unknown-elf"
    else:
        target_flag = "--target=aie2-none-unknown-elf"

    kernel_sources = resolve_kernel_sources(info)
    if not kernel_sources:
        return

    for obj_name, src_path in kernel_sources:
        out_obj = build_dir / f"{obj_name}.o"
        stamp = out_obj.with_suffix(out_obj.suffix + ".stamp.json")
        ensure_dir(out_obj.parent)

        stamp_data = {
            "kind": "kernel_obj",
            "src": str(src_path),
            "device": cfg.device,
            "target": target_flag,
            "int_bit_width": cfg.int_bit_width,
            "clangpp": str(clangpp),
        }

        if _up_to_date_with_stamp([out_obj], [src_path], stamp, stamp_data):
            continue

        cmd = [
            clangpp,
            "-O2",
            "-std=c++20",
            target_flag,
            *warning_flags,
            "-DNDEBUG",
            f"-I{mlir_aie_root / 'include'}",
            f"-DBIT_WIDTH={cfg.int_bit_width}",
            "-c",
            str(src_path),
            "-o",
            str(out_obj),
        ]

        run_checked(cmd, cwd=cfg.example_dir)
        _write_json_atomic(stamp, stamp_data)


def resolve_generator_script(cfg: BuildConfig, info: ExampleMakeInfo) -> Path:
    if cfg.generator_script:
        p = cfg.generator_script
        if not p.is_absolute():
            p = (cfg.example_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Generator script not found: {p}")
        return p

    if info.aie_py_src:
        cand = Path(info.aie_py_src)
        cand = cand.resolve() if cand.is_absolute() else (cfg.example_dir / cand).resolve()
        if cand.exists():
            if cfg.placed:
                placed_cand = cand.with_name(f"{cand.stem}_placed{cand.suffix}")
                if placed_cand.exists():
                    return placed_cand
            return cand

    py_name = f"{cfg.targetname}.py"
    if cfg.placed:
        placed_name = f"{cfg.targetname}_placed.py"
        if (cfg.example_dir / placed_name).exists():
            py_name = placed_name
    py_path = (cfg.example_dir / py_name).resolve()
    if py_path.exists():
        return py_path

    # Fallback: if there's only one plausible generator script, use it.
    candidates = [
        p
        for p in sorted(cfg.example_dir.glob("*.py"))
        if p.name not in {"test.py"} and not p.name.startswith("run_")
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        raise FileNotFoundError(
            "Could not determine generator script. Candidates:\n"
            + "\n".join(f"  - {p.name}" for p in candidates)
        )
    raise FileNotFoundError(f"No python generator script found in: {cfg.example_dir}")


def generate_aie_mlir(cfg: BuildConfig, info: ExampleMakeInfo, build_dir: Path, trace: bool) -> Path:
    data_size = cfg.in1_size
    suffix = "trace_" if trace else ""
    out_path = build_dir / f"aie_{suffix}{data_size}.mlir"
    stamp = out_path.with_suffix(out_path.suffix + ".stamp.json")
    ensure_dir(out_path.parent)

    py_path = resolve_generator_script(cfg, info)

    # The MLIR depends on generator args; using only mtimes can silently reuse stale MLIR.
    stamp_data_flags = {
        "kind": "generator_mlir",
        "mode": "flags",
        "script": str(py_path),
        "device": cfg.device,
        "col": cfg.col,
        "in1_size": cfg.in1_size,
        "in2_size": cfg.in2_size,
        "out_size": cfg.out_size,
        "int_bit_width": cfg.int_bit_width,
        "trace": bool(trace),
        "trace_size": cfg.trace_size,
        "extra_args": list(cfg.extra_generator_args),
    }

    if _up_to_date_with_stamp([out_path], [py_path], stamp, stamp_data_flags):
        print(f"[gen] Up-to-date: {out_path.name}")
        return out_path

    base_cmd = [
        sys.executable,
        str(py_path),
        "-d",
        cfg.device,
        "-i1s",
        str(cfg.in1_size),
        "-i2s",
        str(cfg.in2_size),
        "-os",
        str(cfg.out_size),
        "-bw",
        str(cfg.int_bit_width),
    ]
    if trace:
        base_cmd += ["-t", str(cfg.trace_size)]
    if cfg.extra_generator_args:
        base_cmd += cfg.extra_generator_args

    print(f"[gen] Writing MLIR: {out_path}")

    proc = _run_to_file(base_cmd, cwd=cfg.example_dir, out_path=out_path)
    if proc.returncode == 0:
        _write_json_atomic(stamp, stamp_data_flags)
        return out_path

    err = proc.stderr or ""
    # For older examples, the generator often expects positional args.
    # We only retry when it looks like the script rejected argparse flags.
    if _looks_like_flag_incompatible(err):
        retry_cmds: list[tuple[list[str], dict]] = []

        # Common legacy conventions:
        #   a) script.py <devicename>
        #   b) script.py <devicename> <col>
        cmd1 = [sys.executable, str(py_path), cfg.device]
        stamp1 = dict(stamp_data_flags)
        stamp1["mode"] = "positional_device"
        retry_cmds.append((cmd1, stamp1))

        cmd2 = [sys.executable, str(py_path), cfg.device, str(cfg.col)]
        stamp2 = dict(stamp_data_flags)
        stamp2["mode"] = "positional_device_col"
        retry_cmds.append((cmd2, stamp2))

        for cmd, stamp_data in retry_cmds:
            print("[gen] Retrying generator with positional args.")
            proc2 = _run_to_file(cmd, cwd=cfg.example_dir, out_path=out_path)
            if proc2.returncode == 0:
                print("[gen] NOTE: generator does not accept flags; using positional invocation.")
                _write_json_atomic(stamp, stamp_data)
                return out_path
            err = proc2.stderr or err
            proc = proc2

    if err:
        print("[gen] stderr:")
        print(err.rstrip())
    raise RuntimeError(f"MLIR generator failed (exit={proc.returncode}).")


def build_xclbin_and_insts(
    cfg: BuildConfig,
    info: ExampleMakeInfo,
    build_dir: Path,
    mlir_path: Path,
    trace: bool,
) -> tuple[Path, Path]:
    data_size = cfg.in1_size

    # Most examples generate NPU insts (.bin), but "runlist" style examples generate an ELF.
    if info.uses_elf_insts:
        xclbin_name = info.xclbin_name_hint or "final.xclbin"
        insts_name = info.insts_name_hint or "insts.elf"
    else:
        xclbin_name = f"final_{'trace_' if trace else ''}{data_size}.xclbin"
        insts_name = f"insts_{data_size}.bin"

    xclbin_path = build_dir / xclbin_name
    insts_path = build_dir / insts_name
    stamp = build_dir / f".aiecc_{'trace' if trace else 'run'}_{data_size}.stamp.json"

    ensure_dir(build_dir)

    rel_mlir = os.path.relpath(mlir_path, start=build_dir)

    aiecc_args = [
        "--aie-generate-xclbin",
        "--no-compile-host",
        f"--xclbin-name={xclbin_name}",
    ]

    if info.uses_elf_insts:
        aiecc_args += [
            "--aie-generate-elf",
            f"--elf-name={insts_name}",
        ]
    else:
        aiecc_args += [
            "--aie-generate-npu-insts",
            f"--npu-insts-name={insts_name}",
        ]

    # Default to Peano flow unless the user explicitly asked for Chess.
    if (not cfg.chess) or (cfg.chess and not xchesscc_available()):
        aiecc_args += ["--no-xchesscc", "--no-xbridge"]

    aiecc_args += [rel_mlir]

    cmd = build_aiecc_command(aiecc_args)

    obj_inputs = [build_dir / f"{n}.o" for n in info.kernel_object_names]
    inputs = [mlir_path] + [p for p in obj_inputs if p.exists()]

    stamp_data = {
        "kind": "aiecc",
        "trace": bool(trace),
        "uses_elf_insts": bool(info.uses_elf_insts),
        "aiecc_args": list(aiecc_args),
    }

    if _up_to_date_with_stamp([xclbin_path, insts_path], inputs, stamp, stamp_data):
        print(f"[aiecc] Up-to-date: {xclbin_path.name}, {insts_path.name}")
    else:
        run_checked(cmd, cwd=build_dir)
        _write_json_atomic(stamp, stamp_data)

    # Some examples name outputs differently; fall back to the most recent artifacts.
    if not xclbin_path.exists():
        candidates = sorted(build_dir.glob("*.xclbin"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            xclbin_path = candidates[0]
        else:
            raise FileNotFoundError(f"Expected xclbin not produced: {xclbin_path}")

    if not insts_path.exists():
        patterns = ["insts_*.bin", "*.bin"]
        if info.uses_elf_insts:
            patterns = ["insts*.elf", "*.elf"] + patterns

        inst_candidates: list[Path] = []
        for pat in patterns:
            inst_candidates.extend(build_dir.glob(pat))

        inst_candidates = sorted(set(inst_candidates), key=lambda p: p.stat().st_mtime, reverse=True)
        if inst_candidates:
            insts_path = inst_candidates[0]
        else:
            raise FileNotFoundError(f"Expected insts not produced: {insts_path}")

    return (xclbin_path, insts_path)

def _copy_if_newer(src: Path, dst: Path) -> bool:
    # Copy src -> dst if src is newer (or dst missing). Returns True if copied.
    if dst.exists():
        try:
            if dst.stat().st_mtime >= src.stat().st_mtime:
                return False
        except FileNotFoundError:
            pass
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def build_host_exe(cfg: BuildConfig) -> Path:
    """Build the host executable for this example via CMake.

    We keep this incremental:
      - Configure is skipped if the cached configuration matches the current settings.
      - Build is always invoked (CMake will decide if anything is up-to-date).
    """
    data_size = cfg.in1_size
    target_exe_base = f"{cfg.targetname}_{data_size}"
    host_build_dir = cfg.example_dir / "_build"
    ensure_dir(host_build_dir)

    use_windows_host = is_windows() or is_wsl()

    # Configure (skip if cached config matches).
    configure_stamp = host_build_dir / ".cmake_configure.stamp.json"
    configure_data = {
        "kind": "cmake_configure",
        "schema_version": 2,
        "source_dir": str(cfg.example_dir),
        "target_exe_base": target_exe_base,
        "in1_size": cfg.in1_size,
        "in2_size": cfg.in2_size,
        "out_size": cfg.out_size,
        "int_bit_width": cfg.int_bit_width,
        "generator_request": cfg.generator,
        "generator_effective": _cmake_generator_from_cache(host_build_dir) or cfg.generator,
        "config": cfg.config,
        "platform": ("wsl_windows_host" if is_wsl() else ("windows" if is_windows() else "posix")),
    }

    need_configure = True
    if (host_build_dir / "CMakeCache.txt").exists() and _stamp_matches(configure_stamp, configure_data):
        need_configure = False

    if need_configure:
        if is_wsl():
            # In WSL, the NPU is exposed via the Windows driver/XRT. Build and run the host via Windows tools,
            # using \wsl$ paths so the Windows build can see the sources and artifacts.
            win_src = _wsl_to_windows_path(cfg.example_dir)
            win_build = _wsl_to_windows_path(host_build_dir)
            cmake_configure = [
                "powershell.exe",
                "cmake",
                "-S",
                win_src,
                "-B",
                win_build,
                f"-DTARGET_NAME={target_exe_base}",
                f"-DIN1_SIZE={cfg.in1_size}",
                f"-DIN2_SIZE={cfg.in2_size}",
                f"-DOUT_SIZE={cfg.out_size}",
                f"-DINT_BIT_WIDTH={cfg.int_bit_width}",
            ]
            if cfg.generator:
                cmake_configure += ["-G", cfg.generator]

            requested_gen = cfg.generator or ""
            if not _is_multi_config_generator(requested_gen):
                cmake_configure += [f"-DCMAKE_BUILD_TYPE={cfg.config}"]
        else:
            cmake_configure = [
                "cmake",
                "-S",
                str(cfg.example_dir),
                "-B",
                str(host_build_dir),
                f"-DTARGET_NAME={target_exe_base}",
                f"-DIN1_SIZE={cfg.in1_size}",
                f"-DIN2_SIZE={cfg.in2_size}",
                f"-DOUT_SIZE={cfg.out_size}",
                f"-DINT_BIT_WIDTH={cfg.int_bit_width}",
            ]
            if cfg.generator:
                cmake_configure += ["-G", cfg.generator]

            # Single-config generators want CMAKE_BUILD_TYPE (including NMake/Ninja on Windows).
            # Multi-config generators (VS/Xcode/Ninja Multi-Config) ignore it.
            requested_gen = cfg.generator or ""
            if not _is_multi_config_generator(requested_gen):
                cmake_configure += [f"-DCMAKE_BUILD_TYPE={cfg.config}"]

        run_checked(cmake_configure, cwd=cfg.example_dir)
        _write_json_atomic(configure_stamp, configure_data)

    # Build (incremental).
    used_gen = _cmake_generator_from_cache(host_build_dir) or cfg.generator or ""
    if is_wsl():
        win_build = _wsl_to_windows_path(host_build_dir)
        cmake_build = ["powershell.exe", "cmake", "--build", win_build]
    else:
        cmake_build = ["cmake", "--build", str(host_build_dir)]
    if _is_multi_config_generator(used_gen):
        cmake_build += ["--config", cfg.config]
    run_checked(cmake_build, cwd=cfg.example_dir)

    exe_file = host_exe_name(target_exe_base)

    # Common output locations:
    candidates = [
        host_build_dir / cfg.config / exe_file,   # multi-config (VS/MSBuild)
        host_build_dir / exe_file,                # single-config (Ninja/Unix Makefiles)
    ]

    src_exe: Optional[Path] = None
    for cand in candidates:
        if cand.exists():
            src_exe = cand
            break

    # Fallback: find by name, pick newest (ignore CMake internals).
    if src_exe is None:
        hits: list[Path] = []
        for cand in host_build_dir.rglob(exe_file):
            if any(part in {"CMakeFiles"} or part.endswith(".dir") for part in cand.parts):
                continue
            if cand.is_file():
                hits.append(cand)
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            src_exe = hits[0]

    if src_exe is None:
        raise FileNotFoundError(
            f"Could not find built host executable '{exe_file}'. Searched: {candidates} (+ recursive fallback)"
        )

    out = cfg.example_dir / exe_file
    copied = _copy_if_newer(src_exe, out)
    if copied:
        print(f"[host] Copied host exe -> {out}")
    else:
        print(f"[host] Up-to-date: {out}")
    return out

def run_host_exe(exe: Path, xclbin: Path, insts: Path, trace_size: Optional[int]) -> None:
    if is_wsl():
        # The host binary is a Windows executable (built via Windows CMake) and must be launched via Windows.
        exe_win = _wsl_to_windows_path(exe)
        xclbin_win = _wsl_to_windows_path(xclbin)
        insts_win = _wsl_to_windows_path(insts)

        def _ps_quote(s: str) -> str:
            return "'" + s.replace("'", "''") + "'"

        cmd = (
            f"& {_ps_quote(exe_win)} "
            f"-x {_ps_quote(xclbin_win)} "
            f"-i {_ps_quote(insts_win)} "
            f"-k MLIR_AIE"
        )
        if trace_size is not None:
            cmd += f" -t {trace_size}"
        ps = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
        run_checked(ps, cwd=exe.parent)
        return

    cmd = [str(exe), "-x", str(xclbin), "-i", str(insts), "-k", "MLIR_AIE"]
    if trace_size is not None:
        cmd += ["-t", str(trace_size)]
    run_checked(cmd, cwd=exe.parent)

def run_python_test(example_dir: Path, xclbin: Path, insts: Path, cfg: BuildConfig, trace: bool) -> None:
    test_py = example_dir / "test.py"
    if not test_py.exists():
        raise FileNotFoundError(f"test.py not found: {test_py}")

    cmd = [
        sys.executable,
        str(test_py),
        "-x",
        str(xclbin),
        "-i",
        str(insts),
        "-k",
        "MLIR_AIE",
        "-i1s",
        str(cfg.in1_size),
        "-i2s",
        str(cfg.in2_size),
        "-os",
        str(cfg.out_size),
    ]
    if trace:
        cmd += ["-t", str(cfg.trace_size)]
    run_checked(cmd, cwd=example_dir)


def _resolve_repo_root_for_trace_scripts(example_dir: Path) -> Optional[Path]:
    # Try to find `python/utils/trace` from repo root.
    # 1) Prefer the checkout containing this script.
    try:
        here = Path(__file__).resolve()
        for cand in [here.parent, *here.parents]:
            if (cand / "python" / "utils" / "trace" / "parse.py").exists():
                return cand
    except Exception:
        pass

    # 2) Fall back to scanning example_dir parents.
    for cand in [example_dir] + list(example_dir.parents)[:10]:
        if (cand / "python" / "utils" / "trace" / "parse.py").exists():
            return cand

    return None


def _resolve_trace_scripts(example_dir: Path) -> Optional[tuple[Path, Path]]:
    # Return the (parse.py, get_trace_summary.py) paths if available.
    repo_root = _resolve_repo_root_for_trace_scripts(example_dir)
    if repo_root is None:
        return None

    parse_py = repo_root / "python" / "utils" / "trace" / "parse.py"
    summary_py = repo_root / "python" / "utils" / "trace" / "get_trace_summary.py"
    if not parse_py.exists() or not summary_py.exists():
        return None

    return parse_py, summary_py


def parse_trace_outputs(example_dir: Path, build_dir: Path, cfg: BuildConfig) -> None:
    # Mirrors the Makefile's trace parsing helpers, when available.
    scripts = _resolve_trace_scripts(example_dir)
    if scripts is None:
        print("[trace] NOTE: trace parsing scripts not found; skipping parse/summary.")
        return

    parse_py, summary_py = scripts

    data_size = cfg.in1_size
    aie_trace_mlir = build_dir / f"aie_trace_{data_size}.mlir"
    if not aie_trace_mlir.exists():
        print("[trace] NOTE: aie trace mlir not found; skipping parse/summary.")
        return

    out_json = example_dir / f"trace_{cfg.targetname}.json"

    cmd_parse = [
        sys.executable,
        str(parse_py),
        "--input",
        "trace.txt",
        "--mlir",
        str(aie_trace_mlir),
        "--output",
        str(out_json),
    ]
    run_checked(cmd_parse, cwd=example_dir)

    cmd_summary = [sys.executable, str(summary_py), "--input", str(out_json)]
    run_checked(cmd_summary, cwd=example_dir)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run mlir-aie programming examples (python backend).")

    p.add_argument(
        "action",
        nargs="?",
        default="build",
        choices=["build", "run", "run_py", "trace", "trace_py", "host", "clean"],
        help="What to do (default: build).",
    )

    p.add_argument(
        "--example-dir",
        default=".",
        help="Example directory containing a Makefile (default: cwd).",
    )
    p.add_argument(
        "--targetname",
        default="",
        help="Override Makefile targetname (rare). If empty, it is parsed from the Makefile.",
    )
    p.add_argument(
        "--generator-script",
        default="",
        help="Override the AIE generator python script (default: from aie_py_src/targetname).",
    )
    p.add_argument(
        "--gen-arg",
        action="append",
        default=[],
        help="Extra argument passed to the generator script (repeatable).",
    )
    p.add_argument(
        "--device",
        choices=["npu", "npu2"],
        default=None,
        help="Target device (default: from NPU2 env, else npu2 on Windows / npu on Linux).",
    )
    p.add_argument(
        "--col",
        type=int,
        default=None,
        help="AIE column index (used by some generator scripts and Makefiles; default: from Makefile if present, else 0).",
    )
    p.add_argument(
        "--int-bit-width",
        type=int,
        default=None,
        help="Bit width (default: from Makefile, else 16).",
    )
    p.add_argument(
        "--in1-size",
        type=int,
        default=None,
        help="Input 1 size in bytes (default: inferred from bit width).",
    )
    p.add_argument(
        "--in2-size",
        type=int,
        default=None,
        help="Input 2 size in bytes (default: Makefile, else inferred from bit width).",
    )
    p.add_argument(
        "--out-size",
        type=int,
        default=None,
        help="Output size in bytes (default: equals in1-size).",
    )
    p.add_argument(
        "--trace-size",
        type=int,
        default=None,
        help="Trace buffer size in bytes (default: Makefile, else 8192).",
    )
    p.add_argument(
        "--placed",
        action="store_true",
        help="Prefer <targetname>_placed.py if present.",
    )
    p.add_argument(
        "--chess",
        action="store_true",
        help="Use proprietary Chess/xchesscc compiler if available.",
    )
    p.add_argument(
        "--config",
        default="Release",
        help="CMake build configuration (Release/Debug).",
    )
    p.add_argument(
        "--generator",
        default=None,
        help='CMake generator (e.g. Ninja, "Visual Studio 17 2022"). Default: auto.',
    )

    return p


def _default_device_from_env() -> Optional[str]:
    # Makefiles commonly use:
    #   devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
    v = (os.environ.get("NPU2") or "").strip()
    if v == "1":
        return "npu2"
    if v == "0":
        return "npu"
    return None


def _maybe_warn_trace_target_missing(info: ExampleMakeInfo) -> None:
    if not info.has_trace:
        print("[trace] NOTE: Makefile has no obvious trace target; attempting trace build anyway.")


def _build_config_from_args(args: argparse.Namespace, example_dir: Path, info: ExampleMakeInfo) -> BuildConfig:
    # Defaults
    device = args.device
    if device is None:
        device = _default_device_from_env()
    if device is None:
        device = "npu2" if is_windows() else "npu"

    col = args.col
    if col is None:
        col = info.default_col if info.default_col is not None else 0

    bit_width = args.int_bit_width
    if bit_width is None:
        bit_width = info.default_int_bit_width or 16

    in1_default, in2_default, _out_default = default_sizes_for_bitwidth(bit_width)

    in1 = args.in1_size if args.in1_size is not None else in1_default
    in2 = args.in2_size if args.in2_size is not None else (info.default_in2_size or in2_default)
    out = args.out_size if args.out_size is not None else in1

    trace_size = args.trace_size if args.trace_size is not None else (info.default_trace_size or 8192)

    generator_script = Path(args.generator_script) if args.generator_script else None

    return BuildConfig(
        example_dir=example_dir,
        targetname=info.targetname,
        device=device,
        col=col,
        int_bit_width=bit_width,
        in1_size=in1,
        in2_size=in2,
        out_size=out,
        trace_size=trace_size,
        placed=args.placed,
        chess=args.chess,
        config=args.config,
        generator=args.generator,
        generator_script=generator_script,
        extra_generator_args=args.gen_arg,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    example_dir = Path(args.example_dir).resolve()
    action = args.action

    if action == "clean":
        # Clean does not require the toolchain environment; keep it lightweight.
        build_dir = example_dir / "build"
        host_build_dir = example_dir / "_build"
        safe_rmtree(build_dir)
        safe_rmtree(host_build_dir)

        targetname = args.targetname
        if not targetname:
            try:
                info_for_clean = parse_makefile(example_dir)
                targetname = info_for_clean.targetname
            except Exception:
                targetname = example_dir.name

        # Host executables are commonly named: <targetname>_<data_size>[.exe]
        exe_suffix = ".exe" if is_windows() else ""
        for exe in example_dir.glob(f"{targetname}_*{exe_suffix}"):
            try:
                exe.unlink()
            except Exception:
                pass

        print("[clean] Done.")
        return 0

    preflight_or_die(action)

    info = parse_makefile(example_dir)
    if args.targetname:
        info.targetname = args.targetname

    cfg = _build_config_from_args(args, example_dir, info)
    build_dir = cfg.example_dir / "build"

    is_trace = action in TRACE_ACTIONS

    def _build_aie(trace: bool) -> tuple[Path, Path]:
        ensure_dir(build_dir)
        compile_kernel_objects(cfg, info, build_dir)
        mlir_path = generate_aie_mlir(cfg, info, build_dir, trace=trace)
        return build_xclbin_and_insts(cfg, info, build_dir, mlir_path, trace=trace)

    xclbin: Optional[Path] = None
    insts: Optional[Path] = None

    if action in AIE_ACTIONS:
        if is_trace:
            _maybe_warn_trace_target_missing(info)
        xclbin, insts = _build_aie(trace=is_trace)

    if action == "build":
        print("[done] build")
        return 0

    if action == "host":
        exe = build_host_exe(cfg)
        print(f"[done] host -> {exe}")
        return 0

    exe: Optional[Path] = None
    if action in HOST_BUILD_ACTIONS:
        exe = build_host_exe(cfg)

    if action in RUN_HOST_ACTIONS:
        assert exe is not None and xclbin is not None and insts is not None
        run_host_exe(exe, xclbin, insts, trace_size=cfg.trace_size if is_trace else None)

    if action in RUN_PY_ACTIONS:
        assert xclbin is not None and insts is not None
        run_python_test(cfg.example_dir, xclbin, insts, cfg, trace=is_trace)

    if is_trace:
        parse_trace_outputs(cfg.example_dir, build_dir, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
