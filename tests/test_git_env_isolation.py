"""Tests that create throwaway git repos must not inherit a calling git hook's environment.

A pre-commit hook exports GIT_INDEX_FILE / GIT_DIR etc.; a test running `git commit` in a temp repo would then
write into the outer repository (seen 21 Nov 2026 on prereg/wmm). conftest scrubs these for every test."""

import os

import pytest

LEAKY = (
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_PREFIX",
    "GIT_COMMON_DIR",
)


@pytest.mark.parametrize("name", LEAKY)
def test_git_hook_environment_is_scrubbed(name):
    assert name not in os.environ
