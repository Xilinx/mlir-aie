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

### Documenting Python Code

Please document your Python code with docstrings. For Visual Studio Code, we recommend using a plugin such as
"autoDocstring - Python Docstring Generator" or similar.

## Issues

GitHub issues are used to report bugs and questions. If you create an issue, please make sure it contains enough information to reproduce it.

## License

By contributing, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.