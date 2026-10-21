# Docker images per matplotlib backend (docker bake)

Build three runtime images from one Dockerfile, one per interactive matplotlib
backend, with a single `docker buildx bake` call.

| Image tag | Backend | Display | Extra Python deps |
|---|---|---|---|
| `magnet-scipy:<version>-tkagg` | TkAgg | X11 | none (`python3-tk` from apt) |
| `magnet-scipy:<version>-qtagg` | QtAgg | X11 | `gui` extra: `PySide6-Essentials` |
| `magnet-scipy:<version>-webagg` | WebAgg | browser, port 8988 | `web` extra: `tornado` |

Each image also gets a moving `magnet-scipy:<backend>` tag.

## Decisions

- One `docker/Dockerfile`, shared `build` stage parameterised by
  `ARG UV_EXTRAS`, final stages `tkagg`, `qtagg`, `webagg` on a common `runtime`.
- `docker-bake.hcl` at the repo root, one matrix target, `default` group builds
  all three. `VERSION` comes from the environment (bake cannot read
  `pyproject.toml`), default `dev`. Tag format `<version>-<backend>`.
- Qt binding: `PySide6-Essentials` (LGPL, fits the MIT licence) as an optional
  `gui` extra, so a plain install keeps TkAgg.
- `MPLBACKEND` stays unset in tkagg/qtagg images (auto-detection picks the
  backend with a display, falls back to Agg without). WebAgg is never
  auto-detected, so the webagg image ships a `matplotlibrc` selecting it and
  binding to `0.0.0.0`.
- `uv sync` steps use a shared BuildKit cache mount (`/root/.cache/uv`): the
  first parallel bake without it timed out downloading scipy, as each target
  fetched every wheel on its own.
- Singularity recipe and devcontainer untouched.

## Steps

1. `pyproject.toml`: add `gui` and `web` extras; re-lock `uv.lock`
   → verify: lock contains `pyside6-essentials` and `tornado`.
2. `docker/Dockerfile`: multi-target rewrite; `docker/matplotlibrc.webagg`;
   whitelist it in `docker/Dockerfile.dockerignore`.
3. `docker-bake.hcl` → verify: `docker buildx bake --print` lists 3 targets/tags.
4. `docker/README.md` with build and run instructions.
5. Build all three → verify: `--help` runs in each; Qt xcb plugin has no
   missing libraries (`ldd`); webagg image reports `WebAgg` backend.

## Future

- SCM-derived version (setuptools_scm) would need `SETUPTOOLS_SCM_PRETEND_VERSION`
  passed into the build and `+` replaced in Docker tags.
