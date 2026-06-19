# pyright import-resolution stubs

`aie` is a symlink to `../../python`. The package sources under `python/` import
themselves with the installed package name (`aie.*`) and with package-relative
wildcards (e.g. `from ..dialects.aie import buffer`). Those only resolve when the
source files appear inside a package literally named `aie`.

Pointing pyright's `extraPaths` here (see the repo-root `pyrightconfig.json`) lets it
analyze the source tree in `python/` as if it were the installed `aie` package,
without needing a wheel install of the sources under test.

The generated C-extension bindings and tablegen'd op wrappers (`aie._mlir_libs`,
`aie.ir`, `aie.dialects._*_ops_gen`, `aie.extras`, ...) are not in `python/`; they
come from a build. `pyrightconfig.json` resolves those via the `install/python`
extraPath, so a build/install must exist for a clean run.
