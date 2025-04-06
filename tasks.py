r"""Define some tasks that are executed with invoke."""

from __future__ import annotations

from typing import TYPE_CHECKING

from invoke import task

if TYPE_CHECKING:
    from invoke.context import Context

NAME = "evalix"
SOURCE = f"src/{NAME}"
TESTS = "tests"
UNIT_TESTS = f"{TESTS}/unit"
INTEGRATION_TESTS = f"{TESTS}/integration"


@task
def create_venv(c: Context) -> None:
    r"""Create a virtual environment."""
    c.run("uv venv")
    c.run("source .venv/bin/activate")
    c.run("uv python list --only-installed")
    c.run("uv python find")
    c.run("make install-invoke")


@task
def install(c: Context, all_deps: bool = False) -> None:
    r"""Install packages."""
    cmd = ["uv pip install -r pyproject.toml"]
    if all_deps:
        cmd.append("--all-extras --group dev")
    c.run(" ".join(cmd))
    c.run("uv pip install -e .")


@task
def update(c: Context) -> None:
    r"""Update the dependencies and pre-commit hooks."""
    c.run("uv sync --upgrade --all-extras")
    c.run("uv tool upgrade --all")
    c.run("pre-commit autoupdate")


@task
def unit_tests(c: Context, cov: bool = False) -> None:
    r"""Run the unit tests."""
    cmd = ["python -m pytest --xdoctest --timeout 10"]
    if cov:
        cmd.append(f"--cov-report html --cov-report xml --cov-report term --cov={NAME}")
    cmd.append(f"{UNIT_TESTS}")
    c.run(" ".join(cmd))


@task
def show_installed_packages(c: Context) -> None:
    r"""Show the installed packages."""
    c.run("uv pip list")
    c.run("uv pip check")
