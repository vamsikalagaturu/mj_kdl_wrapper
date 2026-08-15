# AGENTS.md

## Release pipeline

The release PR is merged with a **merge commit**, not a squash. That is what keeps `main` and
`dev` from diverging: a merge makes `dev` an ancestor of `main`, so bringing `main` back is a
fast-forward with nothing to resolve. A squash would put a commit on `main` that `dev` has no
history of, and every later release would open with conflicts in the version files.

Squash is still fine for feature PRs into `dev` -- it is release PRs into `main` that must
merge.

Cutting `vX.Y.Z`, in order:

1. **Check the version.** `cmake/Versions.cmake` (`MJ_KDL_VERSION`) and `pyproject.toml`
   (`version`) must already read `X.Y.Z` -- `dev` carries the *next* version, bumped at step 6
   of the previous release. Nothing to bump here.
2. **PR `dev` -> `main`.** All CI checks must pass: `build`, `test`, `docs`,
   `bindings (3.10/3.11/3.12)`, `colcon (jazzy/lyrical)`. `deploy-docs` reports `skipping` and
   is not required.
3. **Merge it** (`gh pr merge --merge`), never squash.
4. **Tag on `main`.** Check out `main`, fast-forward, then `git tag -a vX.Y.Z` and push the tag.
   Never tag `dev`.
5. **Publish the GitHub release** for that tag. This is what deploys the docs to
   <https://mj-kdl-wrapper.vamsi.sh/> -- `docs.yml` only uploads the Pages artifact when
   `github.event_name == 'release'`.
6. **Fast-forward `dev` to `main`, then bump.** Set both version files to the next patch
   version and commit on `dev`.

If step 6 wants to merge rather than fast-forward, something landed on `main` that did not come
through the release PR -- find out what before resolving anything.
