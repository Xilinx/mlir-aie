#!/usr/bin/env bash
# EXPERIMENT (TEMPORARY -- removed before the split PR is finalized).
#
# Proves the compile/execute split on real Ryzen AI hardware and collects the numbers for the
# #3198 CI/CD discussion. It tests correctness INCLUDING the deliberate-failure paths:
#
#   1. whole == split       : every converted npu-xrt test gives the same verdict whole-test vs
#                             split (compile phase + execute phase).
#   2. fresh-root canary     : after wiping compiled artifacts, execute-only must produce NO positive
#                             outcome for any converted test. A PASS/XPASS here = the execute phase
#                             rode a stale artifact -> hard error.
#   3. no-masking injection  : a mutated kernel (still valid C++, wrong output) must COMPILE cleanly
#                             yet FAIL at EXECUTE in BOTH whole and split -> the split cannot mask a
#                             regression. Reverted afterward.
#   4. timing                : whole (serial capacity-1) vs split (parallel compile + serial execute).
#
# Verdicts are read from lit's `-o` JSON report, never `-s`-suppressed stdout (which prints a line
# only for FAILING tests). Guards FAIL LOUD not only on TOTAL vacuity but on (a) PARTIAL coverage loss
# (an enumerated test silently absent from the report -- lit only WARNS on an unresolved input) and
# (b) a missing positive PASS baseline (comparing two uniformly-broken sides proves nothing).
set -uo pipefail

TREE="${TREE:-$PWD}"
BUILD="${BUILD:-$TREE/build}"
NPU_XRT_SRC="$TREE/test/npu-xrt"
OUT="${OUT:-$TREE/split-experiment-out}"
JOBS_COMPILE="${JOBS_COMPILE:-10}"
JOBS_WHOLE="${JOBS_WHOLE:-12}"
LIT="${LIT:-lit}"
export PATH="$BUILD/bin:${AIEBU_DIR:+$AIEBU_DIR:}$PATH"

[ -n "$OUT" ] && rm -rf "$OUT"; mkdir -p "$OUT"   # fresh OUT: never parse a stale prior-run report

fail=0
summary="$OUT/summary.txt"; : > "$summary"
note(){ echo "::notice::$*"; echo "$*" >> "$summary"; }
bad(){  echo "::error::$*";  echo "ASSERT-FAIL: $*" >> "$summary"; fail=1; }

# Abort safety-net for the fault-injection. Revert via a byte-exact BACKUP, not `git checkout`
# (which would also discard a developer's unrelated uncommitted edits to that file on a shared
# local worktree). $_MUT/$_BAK track the in-flight mutated file + its backup.
_MUT=""; _BAK=""
restore_mut(){ [ -n "$_MUT" ] && [ -f "$_BAK" ] && cp -f "$_BAK" "$_MUT"; rm -f "$_BAK"; _MUT=""; _BAK=""; }
trap 'restore_mut' EXIT

# "<testname> <CODE>" per test from a lit JSON report (-o). Empty if unreadable.
codes(){ python3 - "$1" <<'PY'
import sys, json, re
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
for t in d.get('tests', []):
    n = t['name'].split('::')[-1].strip()
    m = re.match(r'npu-xrt/(.+?)/run\.lit', n)
    if m: n = m.group(1)
    print(n, t['code'])
PY
}
count(){      codes "$1" | awk -v c="$2" '$2==c' | wc -l; }
total(){      codes "$1" | wc -l; }
verdict_of(){ codes "$1" | awk -v n="$2" '$1==n{print $2; exit}'; }  # verdict of the NAMED test
elapsed(){    awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", b-a}'; } # no bc dependency

mapfile -t T < <(cd "$NPU_XRT_SRC" && grep -rl '%npu_run%' . --include=run.lit \
                   | sed 's|^\./||;s|/run\.lit$||' | sort)
note "converted npu-xrt tests: ${#T[@]}"
[ "${#T[@]}" -gt 0 ] || { bad "0 converted tests discovered"; echo "::error::experiment aborted"; exit 1; }
TLIST="$OUT/.tlist"; printf '%s\n' "${T[@]}" > "$TLIST"
# codes_enum/count_enum/total_enum: restrict to the CONVERTED tests only, so a non-converted "bonus"
# test that happens to share a dir (lit may discover more tests than we enumerated) cannot inflate a
# count or produce a false canary PASS.
codes_enum(){ awk 'NR==FNR{keep[$1]=1;next} keep[$1]' "$TLIST" <(codes "$1"); }
count_enum(){ codes_enum "$1" | awk -v c="$2" '$2==c' | wc -l; }
total_enum(){ codes_enum "$1" | wc -l; }
# every enumerated test MUST appear in a phase's report, else its claim was silently never exercised.
assert_coverage(){ # $1 report.json  $2 label
  local present t missing=()
  present="$(codes "$1" | awk '{print $1}')"
  for t in "${T[@]}"; do grep -qxF "$t" <<<"$present" || missing+=("$t"); done
  [ "${#missing[@]}" -eq 0 ] || bad "$2: ${#missing[@]}/${#T[@]} enumerated tests missing from report (silent coverage loss): ${missing[*]}"
}

# every enumerated test must exist under $BUILD, else lit silently drops it.
P=(); miss_build=()
for t in "${T[@]}"; do
  if [ -d "$BUILD/test/npu-xrt/$t" ]; then P+=("$BUILD/test/npu-xrt/$t"); else miss_build+=("$t"); fi
done
[ "${#miss_build[@]}" -eq 0 ] || bad "enumerated tests missing under \$BUILD (build lags source): ${miss_build[*]}"

run_mode(){ # $1 label  $2 AIE_NPU_SPLIT  $3 jobs  $4 outfile
  local t0 t1; t0=$(date +%s.%N)
  AIE_NPU_SPLIT="$2" "$LIT" "${P[@]}" -j"$3" -sv --time-tests -o "$4.json" > "$4" 2>&1
  t1=$(date +%s.%N); elapsed "$t0" "$t1" > "$4.wall"
  note "$1: wall=$(cat "$4.wall")s  $(count "$4.json" PASS) pass / $(count "$4.json" FAIL) fail / $(count "$4.json" UNSUPPORTED) unsupp"
}

wipe_artifacts(){ find "$BUILD/test/npu-xrt" \
  \( -name '*.xclbin' -o -name 'test.exe' -o -name '*insts*.bin' -o -name '*.o' \
     -o -name 'aie_arch.mlir' -o -name '*.prj' -o -name 'ctrlpkt*.bin' \) -prune -exec rm -rf {} + 2>/dev/null; }

echo "##################### [1/4] whole vs split ####################"
run_mode WHOLE   ""        "$JOBS_WHOLE"   "$OUT/whole.log"
run_mode COMPILE compile   "$JOBS_COMPILE" "$OUT/compile.log"
run_mode EXECUTE execute   "$JOBS_WHOLE"   "$OUT/execute.log"
assert_coverage "$OUT/whole.log.json"   "whole"
assert_coverage "$OUT/compile.log.json" "compile"
assert_coverage "$OUT/execute.log.json" "execute"
[ "$(count_enum "$OUT/whole.log.json" PASS)" -gt 0 ] || bad "whole: 0 PASS among converted tests -- device/toolchain broken; parity/canary/injection would be vacuous"

python3 - "$OUT" "$TLIST" >> "$summary" 2>&1 <<'PY'
import sys, re, os, json
out, tlist = sys.argv[1], sys.argv[2]
keep = set(l.strip() for l in open(tlist) if l.strip())
def parse(f):
    d={}
    try: j=json.load(open(os.path.join(out,f)))
    except Exception: return d
    for t in j.get('tests',[]):
        n=t['name'].split('::')[-1].strip()
        m=re.match(r'npu-xrt/(.+?)/run\.lit', n)
        if m: n=m.group(1)
        if n in keep: d[n]=t['code']
    return d
# normalize any code into 3 buckets so a shared TIMEOUT/UNRESOLVED isn't a false MISMATCH,
# while PASS-vs-not-PASS (the distinction that matters) is preserved.
def norm(code):
    if code == 'PASS': return 'PASS'
    if code in ('UNSUPPORTED','-'): return 'UNSUPPORTED'
    return 'NONPASS'
w,c,e=parse('whole.log.json'),parse('compile.log.json'),parse('execute.log.json')
tests=sorted(set(w)|set(c)|set(e)); mm=[]
if not tests:
    print("[compare] ERROR: 0 converted tests parsed -- cannot validate whole==split (vacuous)")
    sys.exit(2)
for t in tests:
    W=w.get(t,'-'); C=c.get(t,'-'); E=e.get(t,'-')
    if 'UNSUPPORTED' in (C,E): S='UNSUPPORTED'
    elif C=='PASS' and E=='PASS': S='PASS'
    else: S='NONPASS'
    if norm(W) != S:
        mm.append((t,W,C,E,S))
print(f"[compare] {len(tests)} converted tests, mismatches whole-vs-split: {len(mm)}")
for t,W,C,E,S in mm: print(f"  MISMATCH {t}: whole={W} compile={C} execute={E} -> split={S}")
sys.exit(1 if mm else 0)
PY
[ $? -eq 0 ] || bad "whole != split for one or more converted tests (see compare above)"

echo "##################### [2/4] fresh-root canary #################"
wipe_artifacts
run_mode CANARY-EXECUTE-ONLY execute "$JOBS_WHOLE" "$OUT/canary.log"
assert_coverage "$OUT/canary.log.json" "canary"
ct=$(total_enum "$OUT/canary.log.json")
[ "$ct" -gt 0 ] || bad "canary: 0 converted tests in report (would be vacuous)"
neg=$(( $(count_enum "$OUT/canary.log.json" FAIL) + $(count_enum "$OUT/canary.log.json" UNSUPPORTED) \
       + $(count_enum "$OUT/canary.log.json" UNRESOLVED) + $(count_enum "$OUT/canary.log.json" TIMEOUT) ))
pos=$(( ct - neg ))
if [ "$pos" -ne 0 ]; then bad "canary: $pos converted test(s) produced a POSITIVE outcome execute-only on a wiped root (stale-artifact ride)"
else note "canary OK: 0/$ct converted tests passed execute-only on a wiped root (no stale ride)"; fi

echo "##################### [3/4] no-masking fault injection ########"
inject(){ # $1 test  $2 kernel  $3 sed-good  $4 sed-bad
  local KF="$NPU_XRT_SRC/$1/$2" BP="$BUILD/test/npu-xrt/$1"
  [ -d "$BP" ] || { bad "inject $1: $BP not present under \$BUILD"; return; }
  _MUT="$KF"; _BAK="$OUT/.inj_backup"; cp -f "$KF" "$_BAK"
  sed -i "s|$3|$4|g" "$KF"
  cmp -s "$_BAK" "$KF" && { bad "inject $1: mutation was a no-op (sed matched nothing -- source reformatted?)"; restore_mut; return; }
  local wl="$OUT/inj_${1}_whole.log" sc="$OUT/inj_${1}_split_c.log" se="$OUT/inj_${1}_split_e.log"
  AIE_NPU_SPLIT=""        "$LIT" "$BP" -j1 -sv -o "$wl.json" > "$wl" 2>&1
  AIE_NPU_SPLIT=compile   "$LIT" "$BP" -j1 -sv -o "$sc.json" > "$sc" 2>&1
  AIE_NPU_SPLIT=execute   "$LIT" "$BP" -j1 -sv -o "$se.json" > "$se" 2>&1
  restore_mut
  local wv cv ev; wv=$(verdict_of "$wl.json" "$1"); cv=$(verdict_of "$sc.json" "$1"); ev=$(verdict_of "$se.json" "$1")
  [ -n "$wv" ] && [ -n "$cv" ] && [ -n "$ev" ] || { bad "inject $1: a verdict is missing (0 tests parsed -- would be vacuous)"; return; }
  note "inject $1: whole=$wv split(compile=$cv,execute=$ev)"
  # STRONG: mutated kernel must COMPILE cleanly (cv=PASS) yet FAIL at EXECUTE (ev=FAIL) in both paths,
  # so a FAIL can't be a compile-error masquerading as an execute-layer catch.
  { [ "$cv" = PASS ] && [ "$ev" = FAIL ] && [ "$wv" = FAIL ]; } \
    || bad "inject $1: expected compile=PASS execute=FAIL whole=FAIL (got cv=$cv ev=$ev wv=$wv) -- fault not caught at execute layer, or compile-errored"
}
inject add_one_func_link_with_peano       add_one_kernel.cc "in\[i\] + 1" "in\[i\] + 2"
inject add_one_scale_func_link_with_peano add_one_kernel.cc "in\[i\] + 1" "in\[i\] + 2"

echo "##################### [4/4] timings + summary #################"
{ echo "=== timing (s) ==="
  echo "whole   = $(cat "$OUT/whole.log.wall" 2>/dev/null)"
  echo "compile = $(cat "$OUT/compile.log.wall" 2>/dev/null)  (parallel -j$JOBS_COMPILE)"
  echo "execute = $(cat "$OUT/execute.log.wall" 2>/dev/null)  (serial device)"
} | tee -a "$summary"

echo "===== RESULT ====="; cat "$summary"
[ "$fail" -eq 0 ] && { note "ALL ASSERTIONS PASSED (incl. deliberate fails)"; exit 0; } \
                   || { echo "::error::experiment assertions FAILED"; exit 1; }
