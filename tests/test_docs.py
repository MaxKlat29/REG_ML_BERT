"""Tests for project documentation files."""

import os


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
