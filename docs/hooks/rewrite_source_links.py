# rewrite_source_links.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""MkDocs hook: rewrite relative links that point at repo source into GitHub URLs.

The docs markdown deliberately uses *relative* links to source files and
out-of-tree directories, e.g. ``[worker.py](../../python/iron/worker.py)`` or
``[passthrough](../../programming_examples/basic/passthrough_kernel/)``. Those
links resolve correctly when browsing the repository on GitHub (at any
branch/tag), so the markdown source must stay relative and must NOT be edited.

But those files are not copied into the built MkDocs site, so on the published
site the same links would 404. This hook runs at *build time* (via the
``on_page_markdown`` event) and rewrites only the links that escape the doc
tree into repo source, turning them into absolute ``github.com`` URLs. Links
between doc pages (``docs/`` and ``programming_guide/`` markdown) are left
untouched so they keep resolving locally on the site.

Net effect: GitHub browsing stays correct (source unchanged) AND the site
ships no source-file 404s.
"""

import os
import re
import urllib.parse

# --- Configuration -----------------------------------------------------------
# These may need updating if the project moves to a different GitHub org, or if
# per-version docs should point at a release *tag* instead of ``main`` (in that
# case set GITHUB_BRANCH to the tag, e.g. via an env var at deploy time).
GITHUB_REPO = "Xilinx/mlir-aie"
GITHUB_BRANCH = "main"

_BLOB_BASE = f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}"
_TREE_BASE = f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}"

# Markdown inline link: [text](target)  and  [text](target "title")
# We capture the label, the target, and any optional title, and only touch the
# target. Reference-style links and autolinks are intentionally not handled;
# the docs use inline links for source references.
_LINK_RE = re.compile(
    r"(?P<label>\[(?:[^\[\]]|\[[^\[\]]*\])*\])"  # [text] (allows one nesting level)
    r"\("
    r"\s*"
    r"(?P<target><[^>]*>|[^\s()]+)"  # <target> or bare target (no spaces/parens)
    r"(?P<title>\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?"  # optional "title"
    r"\s*"
    r"\)"
)

# Schemes / prefixes that are never rewritten.
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "//", "tel:")

# Well-known files that carry no extension. Without this, they'd be mistaken
# for directories (GitHub redirects tree->blob anyway, but blob is correct).
_EXTENSIONLESS_FILES = frozenset(
    {
        "Makefile",
        "Dockerfile",
        "CMakeLists.txt",  # has ext, but listed for clarity — handled by splitext
        "LICENSE",
        "README",
        "Kconfig",
        "Gemfile",
    }
)


def _split_suffix(target):
    """Split a link target into (path, suffix) where suffix keeps #anchor/?query.

    Returns the path part and the trailing ``#...`` and/or ``?...`` verbatim so
    it can be re-attached after rewriting the path.
    """
    # Query first, then fragment — but preserve original ordering by scanning
    # for whichever delimiter comes first.
    idx_hash = target.find("#")
    idx_q = target.find("?")
    cut = min([i for i in (idx_hash, idx_q) if i != -1], default=-1)
    if cut == -1:
        return target, ""
    return target[:cut], target[cut:]


def _looks_like_directory(path):
    """Heuristic: does this repo-relative path point at a directory?

    Directories either end in ``/`` or have no file extension on the last
    component. A last component like ``runtime.py`` or ``README.md`` is a file;
    ``passthrough_kernel`` or ``single_core`` is a directory.
    """
    if path.endswith("/"):
        return True
    last = path.rsplit("/", 1)[-1]
    if last in _EXTENSIONLESS_FILES:
        return False
    # A leading-dot-only name (e.g. ".gitignore") still has an extension via
    # splitext returning ('', '.gitignore'); treat presence of a non-empty ext
    # after the name as "file".
    _, ext = os.path.splitext(last)
    return ext == ""


def _rewrite_target(target, page_dir_link, page_dir_real, docs_dir, repo_root):
    """Return the rewritten URL for ``target`` or None to leave it unchanged.

    Two directories for the page are needed because ``docs/programming_guide``
    is a symlink to the real ``<repo>/programming_guide``:

    * ``page_dir_link`` — the page's directory as MkDocs sees it, i.e. the
      *symlink* path under ``docs/`` (from ``page.file.abs_src_path``). Used
      for doc-page detection, matching how MkDocs itself resolves links.
    * ``page_dir_real`` — the symlink-resolved directory. Used to compute the
      real repo-root-relative path for the GitHub URL.

    ``docs_dir`` is the absolute ``docs/`` directory; ``repo_root`` its parent.
    """
    stripped = target.strip()

    # Angle-bracket form <target>
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1].strip()

    if not stripped:
        return None

    # Skip external URLs, protocol-relative, and mail/tel links.
    if stripped.lower().startswith(_SKIP_PREFIXES):
        return None

    # Skip pure anchors and pure queries (same-page).
    if stripped.startswith("#") or stripped.startswith("?"):
        return None

    # Skip already-absolute filesystem paths (site-absolute links).
    if stripped.startswith("/"):
        return None

    # Separate the path from any #anchor / ?query suffix.
    path_part, suffix = _split_suffix(stripped)
    if not path_part:
        return None

    # Decode percent-encoding (e.g. %20) before filesystem resolution, so a
    # target like ``../a%20b/file.py`` resolves against the real path on disk.
    decoded_path = urllib.parse.unquote(path_part)

    # --- Doc-page detection (MkDocs / symlink view) --------------------------
    # Resolve the link lexically against the page's docs-relative directory,
    # exactly as MkDocs does. If that lands on an existing path under docs/,
    # it's a doc page or asset (e.g. ../getting-started.md, ../section-2c/,
    # ../CONTRIBUTING.md) and must stay a working relative link on the site.
    doc_candidate = os.path.normpath(os.path.join(page_dir_link, decoded_path))
    in_docs_lexically = doc_candidate == docs_dir or doc_candidate.startswith(
        docs_dir + os.sep
    )
    if in_docs_lexically and os.path.exists(doc_candidate):
        return None

    # --- Repo-source resolution (real-filesystem / GitHub view) --------------
    # Follow the docs/programming_guide symlink back to the real tree, then
    # apply the relative path — this reproduces how the link resolves when the
    # file is browsed on GitHub.
    resolved = os.path.realpath(os.path.join(page_dir_real, decoded_path))

    real_docs = os.path.realpath(docs_dir)
    real_pg = os.path.realpath(os.path.join(repo_root, "programming_guide"))
    # Belt-and-suspenders: anything that still resolves inside the doc tree
    # (e.g. an intra-programming_guide link, or a broken doc link that doesn't
    # exist on disk) is a doc reference — leave it alone.
    for root in (real_docs, real_pg):
        if resolved == root or resolved.startswith(root + os.sep):
            return None

    # Must live under the repo root to be a source link we can map to GitHub.
    if not (resolved == repo_root or resolved.startswith(repo_root + os.sep)):
        return None

    rel = os.path.relpath(resolved, repo_root)
    # Guard: relpath can produce ".." if something resolved above repo root.
    if rel.startswith(".."):
        return None
    rel = rel.replace(os.sep, "/")

    # Decide blob (file) vs tree (directory). Use the original path_part so a
    # trailing "/" is honored even though realpath strips it.
    is_dir = _looks_like_directory(path_part)
    base = _TREE_BASE if is_dir else _BLOB_BASE

    # Re-encode path segments so spaces etc. are URL-safe, but keep "/" as the
    # separator.
    encoded_rel = "/".join(
        urllib.parse.quote(seg) for seg in rel.split("/")
    )

    return f"{base}/{encoded_rel}{suffix}"


# Track distinct rewritten targets across the whole build for reporting.
_rewritten_targets = set()


def on_page_markdown(markdown, page, config, files):
    """MkDocs event: rewrite repo-source links to GitHub URLs at build time."""
    abs_src = page.file.abs_src_path
    if not abs_src:
        return markdown

    # MkDocs keeps abs_src_path as the (unresolved) path under docs_dir, i.e.
    # docs/programming_guide/... via the symlink. Keep that for doc-page
    # detection; realpath it separately for repo-source mapping.
    page_dir_link = os.path.dirname(os.path.abspath(abs_src))
    page_dir_real = os.path.dirname(os.path.realpath(abs_src))
    docs_dir = os.path.abspath(config["docs_dir"])
    repo_root = os.path.dirname(os.path.realpath(docs_dir))

    def _sub(match):
        target = match.group("target")
        new_target = _rewrite_target(
            target, page_dir_link, page_dir_real, docs_dir, repo_root
        )
        if new_target is None:
            return match.group(0)
        _rewritten_targets.add(new_target)
        label = match.group("label")
        title = match.group("title") or ""
        return f"{label}({new_target}{title})"

    return _LINK_RE.sub(_sub, markdown)


def on_post_build(config):
    """Log rewrite count and drop this hook's copy from the built site.

    Because the hook lives under ``docs_dir`` (``docs/hooks/``), MkDocs copies
    it into ``site/hooks/`` as a static asset. It is build tooling, not site
    content, so we prune it here — keeping mkdocs.yml changes to the single
    ``hooks:`` line rather than adding an ``exclude_docs`` entry.
    """
    import shutil

    try:
        from mkdocs.plugins import log

        log.info(
            "rewrite_source_links: rewrote %d distinct repo-source link target(s) "
            "to github.com/%s @ %s",
            len(_rewritten_targets),
            GITHUB_REPO,
            GITHUB_BRANCH,
        )
    except Exception:
        pass

    try:
        shipped_dir = os.path.join(config["site_dir"], "hooks")
        shipped_file = os.path.join(shipped_dir, os.path.basename(__file__))
        if os.path.isfile(shipped_file):
            os.remove(shipped_file)
        pycache = os.path.join(shipped_dir, "__pycache__")
        if os.path.isdir(pycache):
            shutil.rmtree(pycache, ignore_errors=True)
        # Remove the hooks dir only if it is now empty (don't clobber other
        # legitimately-shipped assets that may live under docs/hooks/).
        if os.path.isdir(shipped_dir) and not os.listdir(shipped_dir):
            os.rmdir(shipped_dir)
    except Exception:
        pass
