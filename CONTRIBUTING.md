<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Contributing

AMD values and encourages community contributions to IRON / MLIR-AIE — bug
reports, questions, docs, and code are all welcome. Please review the guidance
below before contributing.

## Development workflow

We use GitHub to host code, collaborate, and manage version control. All changes
go through pull requests; [GitHub issues](https://github.com/Xilinx/mlir-aie/issues)
track known bugs, and [GitHub Discussions](https://github.com/Xilinx/mlir-aie/discussions)
are the place for usage questions and feature ideas.

## Issue tracking

Before filing a new issue, search the [existing issues](https://github.com/Xilinx/mlir-aie/issues)
to make sure it isn't already listed.

- If your issue already exists, add a comment with any extra detail (such as how
  you reproduced it) rather than opening a duplicate.
- If you're not sure whether it's the same, err on the side of filing and link
  the similar issue by number — we'll close duplicates if they turn out to match.
- Otherwise, open a new issue with the bug-report or feature-request
  [template](https://github.com/Xilinx/mlir-aie/issues/new/choose). Include as
  much detail as you can — what you ran, what you expected, what happened, and
  your environment (device, OS, IRON version) — so we can reproduce it quickly.
  Check back on the issue, since we may need more information.

## Pull requests

1. **Fork** the repo and create a branch off `main` — the default integration
   branch.
2. **Set up the dev environment** so the formatting and lint hooks are in place:
   ```shell
   source utils/env_install.sh --dev
   ```
   This installs the toolchain and registers the pre-commit / pre-push hooks
   (see [Formatting and hooks](#formatting-and-hooks) below).
3. **Make your change**, targeting `main`. Keep the PR focused, and make sure it
   builds.
4. **Add tests for new functionality.** New features should come with a test or
   example so we can confirm they work and stay working. Don't break existing
   tests.
5. **Open the PR** using the
   [pull request template](https://github.com/Xilinx/mlir-aie/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
   Fill in the description and checklist, and link the related issue with
   `Fixes #123`.
6. **Get CI green and work with your reviewer.** CI runs builds, tests, linting,
   and type checks. A maintainer reviews once it passes; push follow-up commits
   to the same branch to address feedback. We'll let you know once your change
   is merged.

Small, self-contained PRs with a clear description are the easiest to review and
the fastest to merge.

By creating a PR, you agree that your contribution will be licensed under the
terms of the [LICENSE](https://github.com/Xilinx/mlir-aie/blob/main/LICENSE) file
in the root of this repository.

## New feature development

For larger features, start a [Discussion](https://github.com/Xilinx/mlir-aie/discussions)
before writing code — maintainers are happy to give direction and feedback early.
Any new feature or API should also come with the corresponding documentation
update (this repository holds its own docs; there is no separate docs repo).

## Formatting and hooks

Formatting is enforced by [pre-commit](https://pre-commit.com/), installed by
`utils/env_install.sh --dev`. The hooks run automatically — validators on every
commit, and the formatters on `git push` — so CI should never be the first place
you find out about a formatting issue. To run them by hand:

```shell
pre-commit run --all-files
```

The hooks cover:

- **C++** — [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html)
  (LLVM style; config in `.clang-format`).
- **Python and notebooks** — [`black`](https://black.readthedocs.io/), plus
  `nbstripout` to scrub notebook output before it is committed.
- **Baseline hygiene** — trailing whitespace, end-of-file, merge-conflict
  markers, and [REUSE](https://reuse.software/) license-header compliance.

If you would rather not install the hooks, you can run `clang-format -i <file>`
and `black <file>` directly, but the hooks are the supported path.

## Type checking Python

The pure-Python package (`python/{iron,utils,helpers,compiler}`) is type-checked
with [pyright](https://github.com/microsoft/pyright) in `standard` mode, and CI
fails on any error. Configuration lives in `pyrightconfig.json` at the repo root.
After a normal build/install you can run:

```shell
pyright
```

When a diagnostic points at a symbol that genuinely exists at runtime but comes
from a compiled extension (`aie._mlir_libs`) or a tablegen-generated op/enum
wildcard (`from aie.dialects... import *`), suppress just that line, naming the
exact rule:

```python
with Context() as ctx:  # pyright: ignore[reportUndefinedVariable]
```

Reserve suppressions for these binding gaps. Fix real type issues at the source
(add a `None` guard or `raise`, tighten an annotation, initialize before a
branch) rather than silencing them. Do not add blanket file-level ignores or
disable rules.

## Documenting your code

- **Python** — document public functions, classes, and modules with docstrings.
  These are rendered into the [API reference](docs/api/index.md) via
  mkdocstrings, so a good docstring is also good published documentation.
- **C++** — use Doxygen-style triple-slash comments (`///`, with `\brief`,
  `\param`, `\returns` as needed) on public declarations in headers. These feed
  the [C++ API reference](docs/api/cpp_doxygen.md).

## Governance

IRON / MLIR-AIE is led and managed by AMD, and we welcome community
contributions. Maintainers (listed in the repository's
[CODEOWNERS](https://github.com/Xilinx/mlir-aie/blob/main/.github/CODEOWNERS)
file) review all proposed changes and can merge to the repository. Everyone else
is a contributor — file issues, improve docs, submit PRs for contained changes,
and propose larger features in Discussions.

By contributing, you agree that your contributions will be licensed under the
LICENSE file in the root of this source tree. Every file must carry an SPDX
license header; the REUSE hook checks this on push.
