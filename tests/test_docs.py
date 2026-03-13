"""Tests for project documentation files and docstring coverage (DOCS-03)."""

import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Any


def test_readme_exists():
    """README.md exists and contains a Mermaid diagram."""
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    assert os.path.isfile(readme_path), "README.md does not exist"
    content = open(readme_path).read()
    assert "```mermaid" in content, "README.md does not contain a Mermaid diagram"


def test_env_example_exists():
    """.env.example exists and contains OPENROUTER_API_KEY."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
    assert os.path.isfile(env_path), ".env.example does not exist"
    content = open(env_path).read()
    assert "OPENROUTER_API_KEY" in content, ".env.example does not contain OPENROUTER_API_KEY"


def _get_src_modules():
    """Return list of (module_path, dotted_name) for all .py files under src/."""
    repo_root = Path(__file__).parent.parent
    src_root = repo_root / "src"
    modules = []
    for py_file in sorted(src_root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        # Convert path to dotted module name
        rel = py_file.relative_to(repo_root)
        dotted = ".".join(rel.with_suffix("").parts)
        modules.append((py_file, dotted))
    return modules


def _has_non_self_params(func_node: ast.FunctionDef) -> bool:
    """Return True if the function has parameters beyond self/cls."""
    args = func_node.args
    all_args = list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)
    non_self = [a for a in all_args if a.arg not in ("self", "cls")]
    # Also count *args and **kwargs
    has_vararg = args.vararg is not None
    has_kwarg = args.kwarg is not None
    return bool(non_self) or has_vararg or has_kwarg


def _has_non_none_return(func_node: ast.FunctionDef) -> bool:
    """Return True if the function has a return annotation that is not None."""
    ann = func_node.returns
    if ann is None:
        return True  # no annotation — be conservative and expect Returns:
    # Check if annotation is None (ast.Constant with value None)
    if isinstance(ann, ast.Constant) and ann.value is None:
        return False
    # Check if annotation is the name "None"
    if isinstance(ann, ast.Name) and ann.id == "None":
        return False
    return True


def test_google_style_docstrings():
    """All public classes and methods across src/ have Google-style docstrings.

    Checks:
      - Each public class (not starting with _) has a docstring.
      - Each public method/function (not starting with _) at module or class
        level has a docstring.
      - If the function has parameters beyond self/cls, docstring contains "Args:".
      - Docstring contains "Returns:" OR function return type annotation is None.

    Nested functions (closures defined inside other functions) are excluded.
    """
    repo_root = Path(__file__).parent.parent
    src_root = repo_root / "src"

    violations: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError) as e:
            violations.append(f"{py_file}: parse error — {e}")
            continue

        rel_path = py_file.relative_to(repo_root)

        # Walk only top-level and class-level nodes (not nested functions)
        def check_node(node: ast.AST, parent_is_class: bool = False):
            """Check a function/class node for docstring completeness."""
            if isinstance(node, ast.ClassDef):
                name = node.name
                if name.startswith("_"):
                    return
                doc = ast.get_docstring(node)
                if not doc:
                    violations.append(
                        f"{rel_path}:{node.lineno}: class {name!r} is missing docstring"
                    )
                # Check methods within class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        check_node(item, parent_is_class=True)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_"):
                    return
                doc = ast.get_docstring(node)
                location = f"{rel_path}:{node.lineno}: def {name!r}"

                if not doc:
                    violations.append(f"{location} is missing docstring")
                    return

                # Args: check
                if _has_non_self_params(node) and "Args:" not in doc:
                    violations.append(
                        f"{location} has parameters but docstring is missing 'Args:' section"
                    )

                # Returns: check (only for non-None return functions)
                if _has_non_none_return(node) and "Returns:" not in doc:
                    violations.append(
                        f"{location} appears to return a value but docstring is missing 'Returns:' section"
                    )

        # Only check module-level and class-level nodes (skip nested functions)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                check_node(node)

    if violations:
        msg = "\n".join(["DOCSTRING VIOLATIONS:"] + [f"  - {v}" for v in violations])
        assert False, msg
