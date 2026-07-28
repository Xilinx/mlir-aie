#!/usr/bin/env bash
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# EXPERIMENT (TEMPORARY -- removed before any cache PR is finalized).
#
# SEPARATE ARM from split_experiment.sh (untouched, independently readable). Proves a
# per-test, content-addressed xclbin cache on real Ryzen AI hardware.
#
# THIS ARM IS MEASUREMENT-ONLY (INTRA-RUN). It demonstrates reuse WITHIN one invocation
# (cold->warm) and proves the cache is SAFE. Genuine cross-*job* persistence (actions/cache)
# is deliberately deferred to a follow-up so the stripped, self-contained version is validated
# on AMD's CI first -- see the commented-out cache steps in zz-experiment-split-validate.yml.
#
#   [A] cold vs warm (INTRA-RUN reuse) : cold = compile every converted npu-xrt test and
#        POPULATE the cache; warm = wipe build artifacts, RESTORE from cache, compile only the
#        misses (~0). NOTE: warm reuses what COLD just wrote in THIS run -- it quantifies the
#        compile time a cache WOULD avoid; it is NOT cross-job persistence (see prehit below).
#   [B] hit is CORRECT : after the warm restore, EXECUTE every test and require the SAME verdict
#        set as cold. A cache hit must be a correct run, not merely a fast one.
#   [C] invalidation self-test (row 4, the load-bearing guard): cache a good kernel -> mutate its
#        source -> assert (1) key CHANGES, (2) cache MISSES, (3) the mutated build COMPILES cleanly
#        then EXECUTES to FAIL (a broken kernel cannot ride a stale hit / mask a regression).
#
# Correctness is read from lit's `-o` JSON report, never `-s`-suppressed stdout. Guards FAIL LOUD
# not only on TOTAL vacuity but on (a) PARTIAL coverage loss (an enumerated test silently missing
# from the report -- lit only WARNS on an unresolved input) and (b) a missing positive PASS baseline
# (comparing two uniformly-broken sides proves nothing).
set -uo pipefail

TREE="${TREE:-$PWD}"
BUILD="${BUILD:-$TREE/build}"
NPU_XRT_SRC="$TREE/test/npu-xrt"
OUT="${OUT:-$TREE/cache-experiment-out}"
# CACHE_DIR MUST live outside $OUT so the start-of-run `rm -rf $OUT` cannot wipe it.
CACHE_DIR="${CACHE_DIR:-$TREE/xclbin-cache-exp}"
JOBS="${JOBS_COMPILE:-10}"
LIT="${LIT:-lit}"
export PATH="$BUILD/bin:${AIEBU_DIR:+$AIEBU_DIR:}$PATH"

[ -n "$OUT" ] && rm -rf "$OUT"; mkdir -p "$OUT" "$CACHE_DIR"   # fresh OUT: never parse a stale prior-run report
find "$CACHE_DIR" -maxdepth 1 -name '*.tmp.*' -exec rm -rf {} + 2>/dev/null || true  # sweep orphaned atomic-save temps

fail=0
summary="$OUT/summary.txt"; : > "$summary"
note(){ echo "::notice::$*"; echo "$*" >> "$summary"; }
bad(){  echo "::error::$*";  echo "ASSERT-FAIL: $*" >> "$summary"; fail=1; }

# Abort safety-net for the row-4 self-test. Revert via a byte-exact BACKUP, not `git checkout`
# (which would also discard a developer's unrelated uncommitted edits to that file on a shared
# local worktree). $_MUT/$_BAK track the in-flight mutated file + its backup.
_MUT=""; _BAK=""
restore_mut(){ [ -n "$_MUT" ] && [ -f "$_BAK" ] && cp -f "$_BAK" "$_MUT"; rm -f "$_BAK"; _MUT=""; _BAK=""; }
trap 'restore_mut' EXIT

# ---- verdicts from lit's JSON report (never from -s-suppressed stdout) ----
codes(){ python3 - "$1" <<'PY'
import sys, json, re
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
for t in d.get('tests', []):
    n=t['name'].split('::')[-1].strip()
    m=re.match(r'npu-xrt/(.+?)/run\.lit', n)
    if m: n=m.group(1)
    print(n, t['code'])
PY
}
count(){      codes "$1" | awk -v c="$2" '$2==c' | wc -l; }
total(){      codes "$1" | wc -l; }
verdict_of(){ codes "$1" | awk -v n="$2" '$1==n{print $2; exit}'; }  # verdict of the NAMED test
elapsed(){    awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", b-a}'; } # no bc dependency
# Every enumerated test MUST appear in a phase's report, else that test's claim was silently
# never exercised (lit only WARNS, not errors, on an input that resolves to no test).
assert_coverage(){ # $1 report.json  $2 label
  local present t missing=()
  present="$(codes "$1" | awk '{print $1}')"
  for t in "${T[@]}"; do grep -qxF "$t" <<<"$present" || missing+=("$t"); done
  [ "${#missing[@]}" -eq 0 ] || bad "$2: ${#missing[@]}/${#T[@]} enumerated tests missing from report (silent coverage loss): ${missing[*]}"
}

# ---- content-addressed keying ----
# GLOBAL fingerprint folded into EVERY per-test key. MUST change whenever codegen could change,
# else a compiler change rides a stale xclbin (a broken pass masked -- worst case for a compiler
# repo). CI sets TOOLCHAIN_KEY=<built commit> (sound SUPERSET; changes every commit). Cross-*commit*
# reuse is intentionally absent (compiler rebuilt per commit); sound reuse is same-commit only.
tc_key(){
  { if [ -n "${TOOLCHAIN_KEY:-}" ]; then printf 'TK=%s\n' "$TOOLCHAIN_KEY"
    else git -C "$TREE" rev-parse HEAD 2>/dev/null
         command -v aiecc >/dev/null && sha256sum "$(command -v aiecc)" 2>/dev/null; fi
    # runtime toolchain (Vitis/XRT) is OS-image state, NOT covered by the commit. Moot for the
    # MEASUREMENT-ONLY (intra-run) experiment (same image within a run), but a FULL XRT/Vitis
    # fingerprint is REQUIRED before re-enabling actions/cache -- fold the Vitis path as a start.
    printf 'VITIS=%s\n' "${VITIS:-none}"
    # shared codegen inputs outside per-test dirs
    find "$TREE/runtime_lib/test_lib" -type f \
         \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cc' \) -print0 2>/dev/null \
      | sort -z | xargs -0 sha256sum 2>/dev/null
  } | sha256sum | awk '{print $1}'
}
TCK="$(tc_key)"; note "global key (toolchain+shared): ${TCK:0:16}..."

# Per-test key = hash(all source files under test/npu-xrt/<t> + global key).
test_key(){ # $1 test
  { find "$NPU_XRT_SRC/$1" -type f \
        \( -name '*.cc' -o -name '*.cpp' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' \
           -o -name '*.mlir' -o -name '*.py' -o -name 'run.lit' -o -name '*.txt' \) -print0 \
      | sort -z | xargs -0 sha256sum 2>/dev/null
    printf 'TCK=%s\n' "$TCK"
  } | sha256sum | awk '{print $1}'
}
slot(){ echo "$CACHE_DIR/$(test_key "$1")"; }

wipe_artifacts(){ find "$BUILD/test/npu-xrt" \
  \( -name '*.xclbin' -o -name 'test.exe' -o -name '*insts*.bin' -o -name '*.o' \
     -o -name 'aie_arch.mlir' -o -name '*.prj' -o -name 'ctrlpkt*.bin' \) -prune -exec rm -rf {} + 2>/dev/null; }
wipe_one(){ [ -d "$BUILD/test/npu-xrt/$1" ] && find "$BUILD/test/npu-xrt/$1" \
  \( -name '*.xclbin' -o -name 'test.exe' -o -name '*insts*.bin' -o -name '*.o' \
     -o -name 'aie_arch.mlir' -o -name '*.prj' -o -name 'ctrlpkt*.bin' \) -prune -exec rm -rf {} + 2>/dev/null; true; }

# cache_save: ATOMIC (temp dir + rename) so an interrupted copy never leaves a partial slot.
# An UNSUPPORTED test never executes, so lit never creates its exec dir -- nothing to cache, not an
# error (realhits, not hits, is what the reuse claim is gated on).
cache_save(){ local s tmp; [ -d "$BUILD/test/npu-xrt/$1" ] || return 1
  s="$(slot "$1")"; tmp="${s}.tmp.$$"
  rm -rf "$tmp"; mkdir -p "$tmp"
  if cp -a "$BUILD/test/npu-xrt/$1/." "$tmp/"; then rm -rf "$s"; mv "$tmp" "$s"
  else rm -rf "$tmp"; return 1; fi; }
# cache_restore: a partial copy is wiped and treated as a MISS (recompile), never a stale ride.
cache_restore(){ local s; s="$(slot "$1")"; [ -d "$s" ] || return 1
  mkdir -p "$BUILD/test/npu-xrt/$1"
  cp -a "$s/." "$BUILD/test/npu-xrt/$1/" || { wipe_one "$1"; return 1; }; }

run_lit(){ # $1 mode(""/compile/execute)  $2 outfile  $3.. test build-paths
  local mode="$1" of="$2"; shift 2
  AIE_NPU_SPLIT="$mode" "$LIT" "$@" -j"$JOBS" -sv --time-tests -o "$of.json" > "$of" 2>&1
}

mapfile -t T < <(cd "$NPU_XRT_SRC" && grep -rl '%npu_run%' . --include=run.lit \
                   | sed 's|^\./||;s|/run\.lit$||' | sort)
note "converted npu-xrt tests: ${#T[@]}"
[ "${#T[@]}" -gt 0 ] || { bad "0 converted tests discovered"; echo "::error::experiment aborted"; exit 1; }
# Feed lit BUILD-tree paths: it resolves each one through test_exec_root back to the source tree,
# so the per-test build dir need NOT exist first (CMake never creates test/npu-xrt/<t>; lit makes it
# on the test's first run). Only a CONFIGURED build is a real precondition; a test that silently
# resolves to nothing is caught downstream by assert_coverage against the JSON report.
[ -f "$BUILD/test/lit.site.cfg.py" ] \
  || { bad "no configured build at \$BUILD ($BUILD/test/lit.site.cfg.py missing)"; echo "::error::experiment aborted"; exit 1; }
P=(); for t in "${T[@]}"; do P+=("$BUILD/test/npu-xrt/$t"); done

echo "##################### [A] cold vs warm (INTRA-RUN reuse) #####################"
prehit=0; for t in "${T[@]}"; do [ -d "$(slot "$t")" ] && prehit=$((prehit+1)); done
note "pre-existing cache slots before this run: ${prehit}/${#T[@]}  (0 expected -- actions/cache is stripped; this arm measures INTRA-run reuse only)"

# COLD: clean root -> compile everything -> execute (reference verdicts) -> populate cache.
wipe_artifacts
t0=$(date +%s.%N); run_lit compile "$OUT/cold_compile.log" "${P[@]}"; t1=$(date +%s.%N)
cold=$(elapsed "$t0" "$t1")
assert_coverage "$OUT/cold_compile.log.json" "cold compile"
run_lit execute "$OUT/cold_execute.log" "${P[@]}"
assert_coverage "$OUT/cold_execute.log.json" "cold execute"
[ "$(count "$OUT/cold_execute.log.json" PASS)" -gt 0 ] || bad "cold execute: 0 PASS -- device/toolchain broken; [A]/[B] would be vacuous"
for t in "${T[@]}"; do cache_save "$t"; done
note "cold: compile-all=${cold}s  ($(count "$OUT/cold_compile.log.json" PASS) pass / $(count "$OUT/cold_compile.log.json" UNSUPPORTED) unsupp)"

# WARM: wipe -> restore from cache -> compile only misses (~0). realhits requires a real device
# artifact so the 20 UNSUPPORTED no-op slots cannot inflate the reuse claim.
wipe_artifacts
hits=0; realhits=0; miss=()
for t in "${T[@]}"; do
  if cache_restore "$t"; then hits=$((hits+1))
     find "$BUILD/test/npu-xrt/$t" \( -name '*.xclbin' -o -name '*insts*.bin' \) 2>/dev/null | grep -q . \
       && realhits=$((realhits+1))
  else miss+=("$t"); fi
done
mp=(); for t in "${miss[@]}"; do mp+=("$BUILD/test/npu-xrt/$t"); done
t0=$(date +%s.%N)
if [ "${#mp[@]}" -gt 0 ]; then run_lit compile "$OUT/warm_compile.log" "${mp[@]}"; else : > "$OUT/warm_compile.log"; fi
t1=$(date +%s.%N); warm=$(elapsed "$t0" "$t1")
note "warm (intra-run): hits=${hits}/${#T[@]} (real compiled-artifact hits=${realhits}; rest unsupported/no-op)  misses=${#miss[@]}  compile-misses=${warm}s"
[ "$realhits" -gt 0 ] || bad "cache: 0 REAL compiled-artifact hits on an unchanged tree -- reuse not working"

echo "##################### [B] cache hit is CORRECT (execute restored) ############"
run_lit execute "$OUT/warm_execute.log" "${P[@]}"
assert_coverage "$OUT/warm_execute.log.json" "warm execute"
[ "$(count "$OUT/warm_execute.log.json" PASS)" -gt 0 ] || bad "warm execute: 0 PASS -- baseline broken; [B] would be vacuous"
python3 - "$OUT" >> "$summary" 2>&1 <<'PY'
import sys, os, json, re
out=sys.argv[1]
def parse(f):
    d={}
    try: j=json.load(open(os.path.join(out,f)))
    except Exception: return d
    for t in j.get('tests',[]):
        n=t['name'].split('::')[-1].strip()
        m=re.match(r'npu-xrt/(.+?)/run\.lit', n)
        if m: n=m.group(1)
        d[n]=t['code']
    return d
cold=parse('cold_execute.log.json'); warm=parse('warm_execute.log.json')
tests=sorted(set(cold)|set(warm)); mm=[t for t in tests if cold.get(t,'-')!=warm.get(t,'-')]
if not tests:
    print("[cache-correct] ERROR: 0 tests parsed -- cannot compare cold vs warm execute (vacuous)"); sys.exit(2)
print(f"[cache-correct] {len(tests)} tests, cold-execute vs warm(restored)-execute mismatches: {len(mm)}")
for t in mm: print(f"  MISMATCH {t}: cold={cold.get(t,'-')} warm={warm.get(t,'-')}")
sys.exit(1 if mm else 0)
PY
[ $? -eq 0 ] || bad "cache hit changed a verdict: restored artifacts do not run identically to a fresh compile"

echo "##################### [C] invalidation self-test (row 4) #####################"
INV_T="add_one_func_link_with_peano"; KF="$NPU_XRT_SRC/$INV_T/add_one_kernel.cc"; BP="$BUILD/test/npu-xrt/$INV_T"
[ -f "$KF" ] || bad "row4: kernel $KF not found (test renamed/moved?)"
# GOOD baseline: compile cleanly + cache + execute PASS.
wipe_one "$INV_T"; run_lit compile "$OUT/inv_good_c.log" "$BP"
[ "$(verdict_of "$OUT/inv_good_c.log.json" "$INV_T")" = PASS ] || bad "row4: good $INV_T did not COMPILE"
cache_save "$INV_T"
run_lit execute "$OUT/inv_good_e.log" "$BP"
[ "$(verdict_of "$OUT/inv_good_e.log.json" "$INV_T")" = PASS ] || bad "row4 precondition: good $INV_T did not EXECUTE-PASS"
k_good="$(test_key "$INV_T")"
# mutate (wrong output). Back up the exact bytes + arm the trap BEFORE touching the file.
_MUT="$KF"; _BAK="$OUT/.kf_backup"; cp -f "$KF" "$_BAK"
sed -i "s|in\[i\] + 1|in\[i\] + 2|g" "$KF"
grep -qF 'in[i] + 2' "$KF" || bad "row4: sed did not match $KF (source reformatted?) -- mutation is a no-op"
k_bad="$(test_key "$INV_T")"
# (1) key must change on mutation
if [ "$k_good" != "$k_bad" ]; then note "row4: key changed on mutation (${k_good:0:12} -> ${k_bad:0:12})"
else bad "row4: content key UNCHANGED after kernel mutation -> cache would ride stale"; fi
# (2) mutated source must MISS the cache
[ -d "$CACHE_DIR/$k_bad" ] && bad "row4: mutated key already present in cache -> stale hit"
# (3) cache-aware build of the mutated test: MISS -> COMPILE cleanly -> EXECUTE to FAIL.
wipe_one "$INV_T"
if cache_restore "$INV_T"; then rode=1; cc="(rode)"
else rode=0; run_lit compile "$OUT/inv_bad_c.log" "$BP"; cc="$(verdict_of "$OUT/inv_bad_c.log.json" "$INV_T")"; fi
run_lit execute "$OUT/inv_bad_e.log" "$BP"
bad_v="$(verdict_of "$OUT/inv_bad_e.log.json" "$INV_T")"
restore_mut   # revert from byte-backup now; EXIT trap is the abort safety-net
[ "$rode" -eq 0 ] || bad "row4: mutated source RESTORED a cached slot (stale ride) -- key did not discriminate"
[ "$cc" = PASS ] || bad "row4: mutated kernel did not COMPILE cleanly (compile=$cc) -- cannot attribute the FAIL to the execute layer"
if [ "$bad_v" = FAIL ]; then note "row4 OK: mutated kernel MISSED cache, compiled, and FAILED on device (no stale ride, no masking)"
else bad "row4: mutated kernel verdict=$bad_v (expected FAIL) -- cache masked a regression"; fi

echo "##################### summary #####################"
{ echo "=== cache reuse (s), INTRA-RUN ==="
  echo "cold compile-all    = $cold"
  echo "warm compile-misses = $warm   (real hits=${realhits}/${#T[@]})"
  echo "cross-JOB reuse (actions/cache): DISABLED this run (stripped); pre-existing slots=${prehit}/${#T[@]}"
} | tee -a "$summary"
echo "===== RESULT ====="; cat "$summary"
[ "$fail" -eq 0 ] && { note "ALL CACHE ASSERTIONS PASSED (incl. invalidation self-test)"; exit 0; } \
                  || { echo "::error::cache experiment assertions FAILED"; exit 1; }
