"""Check that number-producing files match the frozen commit.

Python files are compared by their syntax tree with docstrings removed, so
comment and docstring edits pass and any code change fails. Other files
(configs) are compared byte for byte.

    python frozen_guard.py <commit> <path> [<path> ...]

Exit 0 when everything matches, 1 otherwise; differing files are printed.
"""
import ast
import subprocess
import sys


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_equal(old: bytes, new: bytes, path: str) -> bool:
    if not path.endswith(".py"):
        return old == new
    try:
        a = _strip_docstrings(ast.parse(old))
        b = _strip_docstrings(ast.parse(new))
    except SyntaxError:
        return False
    return ast.dump(a) == ast.dump(b)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True).stdout


def differing(commit: str, paths: list[str]) -> list[str]:
    files = set(_git("ls-tree", "-r", "--name-only", commit, "--", *paths).split())
    files |= set(_git("ls-files", "--", *paths).split())
    bad = []
    for f in sorted(files):
        try:
            old = subprocess.run(["git", "show", f"{commit}:{f}"], check=True,
                                 capture_output=True).stdout
        except subprocess.CalledProcessError:
            bad.append(f + " (new since the frozen commit)")
            continue
        try:
            with open(f, "rb") as fh:
                new = fh.read()
        except FileNotFoundError:
            bad.append(f + " (deleted)")
            continue
        if not code_equal(old, new, f):
            bad.append(f)
    return bad


if __name__ == "__main__":
    bad = differing(sys.argv[1], sys.argv[2:])
    for f in bad:
        print("  " + f)
    sys.exit(1 if bad else 0)
