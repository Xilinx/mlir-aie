# aiecc rebase log — `simplify-aiecc` onto `main`

## Rebase event — 2026-07-07

`simplify-aiecc` was rebased onto the latest `origin/main`. The branch replaces
the old monolithic `tools/aiecc/aiecc.cpp` with the declarative driver, so the
mechanical `aiecc.cpp` conflicts were auto-resolved in favor of the branch
(`git rebase origin/main -X theirs`). Main-side changes to the *old* driver are
therefore **not** carried by git and must be absorbed manually (see the gap list
below and `AIECC_REBASE_GAPS.md`).

## Reference commits

| Ref | Commit | Meaning |
|-----|--------|---------|
| `pre-rebase-simplify-aiecc-20260707` (tag) | `42390e5ab5f` | **Pre-rebase branch tip** (== old `fork/simplify-aiecc`). Snapshot before history was rewritten. |
| `backup/simplify-aiecc-prerebase-20260707` (branch) | `42390e5ab5f` | Same snapshot, as a branch, for convenience. |
| fork point / merge-base | `8da13e91d8a` | Where `simplify-aiecc` originally forked from `main`. |
| new baseline | `a08e167ed40` | `origin/main` tip the branch was rebased onto. |
| new rebased tip | `19a5de5c2c0` | `simplify-aiecc` after the rebase. |

> Recover the exact pre-rebase state at any time with
> `git checkout pre-rebase-simplify-aiecc-20260707` (or the backup branch).

## Feature-gap investigation — old `aiecc` changes on `main`

The old monolithic driver no longer exists on our branch, but its full history
lives on `main`. To review **every** change made to the old aiecc between the
fork point and the rebased baseline:

```bash
# All 14 commits on main that touched tools/aiecc since the fork point:
git log --oneline 8da13e91d8a..a08e167ed40 -- tools/aiecc

# Full diff of the OLD monolithic driver over that range:
git diff 8da13e91d8a a08e167ed40 -- tools/aiecc/aiecc.cpp

# Inspect a single absorbed/unabsorbed commit:
git show <commit> -- tools/aiecc
```

### The 14 old-aiecc commits to reconcile against the declarative driver

| Commit | PR | Summary | Absorption status |
|--------|----|---------|-------------------|
| `1ba5b5cfb6d` | #3261 | Default `-j` to 0 (auto-detect CPU count) | see `AIECC_REBASE_GAPS.md` §5 |
| `cf89594b718` | #3231 | Harmonize license headers + mechanical checks | §7 mechanical |
| `72e91b426cf` | #3179 | Remove dead AIE/AIEX dialect ops/passes | §1 blocker — drop `createAIEObjectFifoRegisterProcessPass()` call |
| `dc50ead8fb8` | #3250 | ThreadPool in `compileCores` | §4 (engine differs) |
| `92e9475ceb4` | #3247 | Downgrade bare inf/nan in phi (Peano) | §2 — port to `downgradeIRForPeano` |
| `0ca8da13438` | #3240 | dynamic-objFifos default | §5 already matched (driver) |
| `fe8f8c22993` | #3243 | scf→cf lowering runtime-sequence-aware | §3 — switch to `createAIESCFToControlFlowPass()` |
| `afc887fcc9e` | #3241 | Downgrade decimal bfloat16 literals (Peano) | §2 — port to `downgradeIRForPeano` |
| `0a69477883a` | #3232 | Downgrade `f0x` float literals (Peano) | §2 — port to `downgradeIRForPeano` |
| `78f5a4390c8` | #3217 | Remove AIEVec→C++ backend | §1 blocker — drop `target-backend=llvmir` token |
| `a2841cb2d22` | #3216 | Per-core compile slices from stripped base | §4 — confirm slice-equivalence test passes |
| `14da22a7d85` | #3214 | Restore align-attribute stripping (Peano) | §2 — port to `downgradeIRForPeano` |
| `7fb70606c97` | #3211 | Lower NPU insts once on full-ELF path | §4 (engine differs) |
| `a0b5ff47372` | #3210 | AMD copyright notices | §7 mechanical |

See `AIECC_REBASE_GAPS.md` for the detailed per-commit actions and the new
`test/aiecc/` tests that arrive with the rebase and must pass.
