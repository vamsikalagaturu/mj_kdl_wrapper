# Git Hooks

This directory contains project-level git hooks.

## Installation

To use these hooks, configure git to find them here:

```
git config core.hooksPath .githooks
```

## Hooks

### pre-commit

Runs `clang-format --dry-run --Werror` on all staged `.cpp` and `.hpp` files.
If any files need formatting, the commit is rejected and the offending files are listed.

To fix: run `clang-format -i <file>` on the listed files, then re-stage and commit.
