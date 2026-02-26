
## 2026-02-20: CI Failure from Missing clang-format

**Mistake**: Committed files that were already staged without running clang-format on ALL files in the commit, causing CI format check failures.

**Root Cause**: 
1. Did not check `git status` carefully before committing - there were pre-staged files from earlier work
2. Only formatted `aiecc.cpp` (the file I edited), not the other staged files

**Prevention Rules**:
1. Before committing, ALWAYS run `git diff --staged --name-only | grep -E '\.(cpp|h)$' | xargs -r clang-format -i` to format ALL staged C++ files
2. After formatting, verify with `clang-format --dry-run -Werror <files>` before committing
3. Check `git status` before committing to understand what will be included
4. When amending commits for CI fixes, use `--force-with-lease` not `--force`

## 2026-02-23: clang-format Version Mismatch

**Mistake**: Local clang-format (v18) produced different output than CI's clang-format (v17), causing format check failures.

**Root Cause**:
1. Different clang-format versions format the same code differently
2. `clang-format --dry-run -Werror` passed locally but failed in CI

**Prevention Rules**:
1. Be aware that CI uses clang-format 17 - local version may differ
2. String concatenations with `<<` should have each string on its own line
3. After rebasing on format-related commits, re-verify formatting
4. When in doubt, format more conservatively (more newlines, not fewer)

## 2026-02-23: Don't Push Without Permission

**Mistake**: Pushed commits without explicit user approval.

**Prevention Rules**:
1. NEVER push unless the user explicitly says "push" or similar
2. After making changes, report what was done and wait for user to approve push
3. Multiple pushes waste CI cycles and can cause confusion

## 2026-02-24: CI uses git clang-format origin/main, not HEAD~1

**Mistake**: Used `git-clang-format --diff HEAD~1` locally which passed, but CI uses `git clang-format origin/main` which failed.

**Root Cause**:
1. CI compares against origin/main, not the previous commit
2. Local clang-format v18 formats differently than CI's clang-format-17
3. The `<<` operator formatting differs between versions - v17 wants each `<<` on its own line

**Prevention Rules**:
1. ALWAYS run `git clang-format origin/main` before pushing, not `git-clang-format HEAD~1`
2. Fetch origin/main first: `git fetch origin main && git clang-format origin/main`
3. When CI fails format check, look at the exact diff in the CI logs and apply those changes
4. clang-format-17 prefers each `<< "string"` on its own line for ostringstream concatenation
5. Do NOT trust local `clang-format --dry-run` if local version differs from CI (v17)
