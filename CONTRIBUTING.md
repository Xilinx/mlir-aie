## Pull Requests

We actively welcome community involvement in this project!

## Developer Guidelines

### Formatting C++

Make sure to format any C++ files you may have modified with:
```shell
clang-format -i <the-file-i-changed>
```

### Formatting Python and Notebooks

For Python, install the formatting packages:
```shell
pip install black
pip install black[jupyter]
```

Make sure to run `black` on all Python and Jupyter notebooks, like so:
```shell
black <the-file-i-changed>
```

The CI will check black formatting of Python and Notebook files; there is a also a commit hook installed
by default in the `quick_setup.sh` process which will not allow you to push until it has scrubbed Jupyter
notebooks of certain information.

### Type Checking Python

The pure-Python package (`python/{iron,utils,helpers,compiler}`) is type-checked with
[pyright](https://github.com/microsoft/pyright) in `standard` mode, and CI fails on any
error. Configuration lives in `pyrightconfig.json` at the repo root.

pyright analyzes the source tree under its installed package name (`aie.*`); the
`aie` symlink in `utils/pyright_stubs/` and the built package at `install/python`
together let it resolve those imports. After a normal build/install you can run:

```shell
pyright
```

When a diagnostic points at a symbol that genuinely exists at runtime but comes from a
compiled extension (`aie._mlir_libs`) or a tablegen-generated op/enum wildcard
(`from aie.dialects... import *`), suppress just that line, naming the exact rule:

```python
with Context() as ctx:  # pyright: ignore[reportUndefinedVariable]
```

Reserve suppressions for these binding gaps. Real type issues should be fixed at the
source (add a `None` guard or `raise`, tighten an annotation, initialize before a
branch) rather than silenced. Do not add blanket file-level ignores or disable rules.

### Documenting Python Code

Please document your Python code with docstrings. For Visual Studio Code, we recommend using a plugin such as
"autoDocstring - Python Docstring Generator" or similar.

## Issues

GitHub issues are used to report bugs and questions. If you create an issue, please make sure it contains enough information to reproduce it.

## License

By contributing, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.