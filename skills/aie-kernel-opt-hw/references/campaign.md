<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Campaigns: many kernels, several agents, one tree

A campaign runs the SKILL.md loop over many kernels at once, with one agent
per group of files, sharing one checkout and one NPU. The per-kernel method
is unchanged. What's new is that your numbers can be contaminated by other
agents' edits and builds. Each rule below cites the incident behind it.

## Before round 1

1. **Survey statically** with `aie-kernel-opt-static`: run remarks over
   every factory, then rank by `unpipelined_loops`, worst `loop/*/II`,
   libcalls and spills:
   ```bash
   python -m aie.utils.compile.remarks --target aie2p --jobs 8 --out $W/survey.json --meta $W/survey-meta.json
   ```
   Assign work **by file**, so each agent owns a disjoint set (F01).
2. **Audit the markers** in the file each factory selects
   (`aie-kernel-opt-static` `static-checks.md` §Resolve; TODO(d-tools:G4)).
   A row with no markers in the selected source reports another kernel. In
   one campaign that happened three times: a `zero.cc` delta was credited to
   matmul and prefill (M05).
3. **Snapshot the base arm** with `git archive`, along with the Python-side
   parameters: stack sizes and geometry tables. A shrunk `_PREFILL_GEOM[256]`
   left `prefill_fv/256` with no before number (F11). Measure the whole matrix
   once from the base, and keep the raw intervals.
4. **Write a brief** that every agent reads. It carries:
   - the setup, gate, remarks, bench and probe commands, with this site's
     `$NPU_LOCK` and `$CPUS`
   - the non-negotiables
   - `aie-kernel-opt-static` `levers.md` and `hw-levers.md`
   - the deliverable format below
   
   Between rounds, add every new measurement trap to the brief (F12).

## Rules in a shared tree

- **Own your files; don't change git state.** No add, commit, stash, checkout
  of a branch, or push. Reverting your own file is fine. The lead commits one
  change per commit, with the numbers in the message (F01).
- **Edit shared files with targeted edits,** never whole-file writes. One
  whole-file write of a factory module silently deleted another agent's hunk
  (F02).
- **Serialize the NPU.** Pick one lock file every agent on the device can
  reach, export it as `NPU_LOCK`, wrap every hardware command in it, and
  keep `-k` selections narrow (F06):
  ```bash
  flock "$NPU_LOCK" pytest test/python/npu/test_kernels_bench.py -m benchmark -k "$CASE" ...
  ```
- **Pin the host side.** Give each agent the same fixed `taskset -c $CPUS`
  list for every run it will compare. Leave out any core the site knows is
  faulty.
- **One `NPU_CACHE_HOME` per agent and per arm,** wiped before each
  reported run (M09).
- **Build your base arm against the live include closure** by copying the
  tree and restoring only your files at the base revision
  (`aie-kernel-opt-static` `static-checks.md` §Arms). `mha.cc` includes `mm.cc`, `softmax.cc` and `zero.cc`, and
  prefill and `mm_fused` include `zero.cc`. A `git archive` arm pins the
  siblings too, so it's only correct while nobody edits them (F05).
- **Source is authoritative for Python.** The JIT reads kernel sources
  through `MLIR_AIE_KERNEL_SOURCES`, but imports factories and cases from the
  installed package. After a Python edit, sync it to the install and build
  trees, then `diff` source against install. An install-only edit once went
  green, and a stale install had 21 differing files (F03).
- Leave `*_scalar` variants alone (F10).
- Keep every `extern "C"` name, signature and buffer layout (C12).

## Provenance

An in-flight fix can erase a finding, but it can never manufacture one. One
of 8 all-clears was clean only because a teammate's uncommitted fix was in
the tree (F04). Next to every number, record:

- the commit
- `git status --short` over the include closure
- a timestamp
- the Peano version
- the source directory each arm compiled from

When two agents disagree, settle it from the object dump, not by picking a
side (F08).

## Deliverable per agent

The headline is **traced cycles per call, base → after, for every case the
kernel appears in**, with the arms run back to back and populations split
with n. Then:

- the static candidate report per change (`aie-kernel-opt-static`)
- the gate command and its pass line
- the mutation that proved the gate
- the proposed commit message
- **every rejected variant, with the number that killed it** (F07)

A well-evidenced NO-CHANGE with its bound is a result (F09).

## Closing

- Re-measure the whole matrix in one session: base arm against the final
  tree.
- Kernels whose bytes didn't change must reproduce to the cycle.
- Check every regression. In one campaign each one was either real and fixed,
  or a mislabelled row.
- Check every number in comments and commit messages against the object and
  the hardware. One commit claimed II84 where the object said 77, and another
  "optimized" an uncalled symbol (F09).
- List what isn't measured: no markers, a base that can't build, entry points
  no case reaches. Say "not measured". Don't estimate.
- Report coverage holes as findings, each with the mutation that proves it.
