# Docker images

`docker/Dockerfile` builds one runtime image per interactive matplotlib
backend. All three share the same build stage and differ only in the
backend-specific packages of the final stage.

| Image | Backend | Plots shown in | Extra Python deps |
|---|---|---|---|
| `magnet-scipy:<version>-tkagg` | TkAgg | X11 window | none (`python3-tk` from apt) |
| `magnet-scipy:<version>-qtagg` | QtAgg | X11 window | `gui` extra (`PySide6-Essentials`) |
| `magnet-scipy:<version>-webagg` | WebAgg | browser, port 8988 | `web` extra (`tornado`) |

Each image is also tagged `magnet-scipy:<backend>`, which always points to the
most recent build of that backend.

## Build

The targets are defined in `docker/docker-bake.hcl`. Run from the repository
root (the build context and Dockerfile paths are resolved relative to the
current directory) and pass the file with `-f`:

```bash
# all three images, in parallel, tagged with the version from pyproject.toml
VERSION=$(python3 -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") \
    docker buildx bake -f docker/docker-bake.hcl

# a single image
docker buildx bake -f docker/docker-bake.hcl magnet-scipy-qtagg

# show the resolved targets, build args and tags without building
docker buildx bake -f docker/docker-bake.hcl --print
```

Without `VERSION` the images are tagged `dev-<backend>`.

With the default builder (`docker` driver) the images land directly in the
local image store. With a `docker-container` builder, add `--load`.

The three targets share a BuildKit cache mount for uv, so wheels are
downloaded once and reused by later builds. `docker builder prune` clears it.

A single image can also be built without bake:

```bash
docker build -f docker/Dockerfile --target qtagg \
    --build-arg UV_EXTRAS="--extra gui" -t magnet-scipy:qtagg .
```

`UV_EXTRAS` must match the target: `""` for `tkagg`, `--extra gui` for
`qtagg`, `--extra web` for `webagg`.

## Run

The current directory is mounted as `/data`, the working directory inside the
container. Without `--show_plots` (or without a display) every image behaves
the same and falls back to the non-interactive Agg backend:

```bash
docker run --rm -v "$PWD":/data magnet-scipy:tkagg magnet-scipy --help
```

### tkagg / qtagg: X11 windows

Forward the X11 socket and authority cookie, and run as your own user so
output files are not owned by root:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
    -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e XAUTHORITY=/tmp/.Xauthority -v "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
    -v "$PWD":/data magnet-scipy:qtagg magnet-scipy-coupled --show_plots ...
```

`MPLBACKEND` is not set in these images: matplotlib picks the image's backend
when `DISPLAY` is reachable and Agg otherwise. Setting `MPLBACKEND` explicitly
would make matplotlib fail instead of falling back when no display is present.

Qt notes:

- If Qt reports `MIT-SHM` / `BadAccess` X errors, add `-e QT_X11_NO_MITSHM=1`
  or `--ipc=host`.
- A warning about `XDG_RUNTIME_DIR` not being set is harmless.

### webagg: plots in the browser

No X11 is needed. The image selects WebAgg through `/etc/matplotlibrc` (copied
from `docker/matplotlibrc.webagg`) and listens on port 8988 on all container
interfaces:

```bash
docker run --rm -it -p 127.0.0.1:8988:8988 --user "$(id -u):$(id -g)" \
    -v "$PWD":/data magnet-scipy:webagg magnet-scipy-coupled --show_plots ...
```

then open <http://localhost:8988>. Publishing on `127.0.0.1` keeps the server
off the local network.

Each `plt.show()` call starts the web server and blocks until you press
Ctrl-C, then the program continues to the next `plt.show()`. Use `-it` so that
Ctrl-C reaches the container.
