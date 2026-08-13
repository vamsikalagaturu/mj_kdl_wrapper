# AGENTS.md

## Release pipeline

`main` is protected and **squash-only**: merging a PR puts one commit on `main`, so `main` and
`dev` diverge by construction at every release. The merge back is not optional tidying -- skip
it and the next release's PR opens with conflicts in the version files and in whatever `main`
squashed away.

Cutting `vX.Y.Z`, in order:

1. **Check the version.** `cmake/Versions.cmake` (`MJ_KDL_VERSION`) and `pyproject.toml`
   (`version`) must already read `X.Y.Z` -- `dev` carries the *next* version, bumped at step 6
   of the previous release. Nothing to bump here.
2. **PR `dev` -> `main`.** All CI checks must pass: `build`, `test`, `docs`,
   `bindings (3.10/3.11/3.12)`, `colcon (jazzy/lyrical)`. `deploy-docs` reports `skipping` and
   is not required.
3. **Squash merge it.** Merge commits and rebase merges are disabled on this repository.
4. **Tag on `main`.** Check out `main`, fast-forward, then `git tag -a vX.Y.Z` and push the tag.
   Never tag `dev`.
5. **Publish the GitHub release** for that tag. This is what deploys the docs to
   <https://mj-kdl-wrapper.vamsi.sh/> -- `docs.yml` only uploads the Pages artifact when
   `github.event_name == 'release'`.
6. **Merge `main` back into `dev`, then bump.** Set both version files to the next patch
   version and commit on `dev`.

Expect the merge in step 6 to conflict on `cmake/Versions.cmake` and `pyproject.toml`: `main`
holds the version just released and `dev` the next one. Keep `dev`'s. Any other conflict is a
real one -- `main` only ever holds what a squash put there.
