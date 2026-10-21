# Runtime images for magnet-scipy, one per interactive matplotlib backend.
#
# Final stages (select with --target): tkagg, qtagg, webagg.
# Build all three from the repository root with docker bake (see docker/README.md):
#   docker buildx bake
# or a single one with plain docker build:
#   docker build -f docker/Dockerfile --target qtagg --build-arg UV_EXTRAS="--extra gui" -t magnet-scipy:qtagg .
# Run (current directory mounted as /data):
#   docker run --rm -v "$PWD":/data magnet-scipy:tkagg magnet-scipy --help
# Run with plot windows (X11 forwarding, tkagg and qtagg images):
#   docker run --rm --user "$(id -u):$(id -g)" \
#       -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -e XAUTHORITY=/tmp/.Xauthority -v "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
#       -v "$PWD":/data magnet-scipy:qtagg magnet-scipy-coupled --show_plots ...
# Run with plots in the browser (webagg image, then open http://localhost:8988):
#   docker run --rm -it -p 127.0.0.1:8988:8988 \
#       -v "$PWD":/data magnet-scipy:webagg magnet-scipy-coupled --show_plots ...

FROM debian:trixie AS build

# Optional dependency groups for the selected backend, e.g. "--extra gui"
ARG UV_EXTRAS=""

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/

# UV_HTTP_TIMEOUT: bake builds the three targets in parallel; the default 30s
# read timeout is too short when they share a slow link.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Install locked dependencies first so this layer is cached across code changes.
# The uv cache mount is shared by the three targets, so wheels are downloaded once.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $UV_EXTRAS

COPY magnet_scipy ./magnet_scipy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $UV_EXTRAS


FROM debian:trixie AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH

WORKDIR /data

CMD ["magnet-scipy", "--help"]


FROM runtime AS tkagg

# python3-tk provides the TkAgg matplotlib backend (X11, no OpenGL needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/*

# MPLBACKEND is left unset: matplotlib picks TkAgg when DISPLAY is set, Agg otherwise
COPY --from=build /opt/venv /opt/venv


FROM runtime AS qtagg

# System libraries for the Qt xcb platform plugin shipped in the PySide6 wheel.
# QtGui links libGL/libEGL even for raster drawing; Mesa's software libraries suffice.
# fonts-dejavu-core gives Qt widgets (toolbar, coordinates) a font.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0t64 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# MPLBACKEND is left unset: matplotlib picks QtAgg when DISPLAY is set, Agg otherwise
COPY --from=build /opt/venv /opt/venv


FROM runtime AS webagg

# WebAgg is never auto-detected: select it and listen on all container interfaces
COPY docker/matplotlibrc.webagg /etc/matplotlibrc
ENV MATPLOTLIBRC=/etc/matplotlibrc

EXPOSE 8988

COPY --from=build /opt/venv /opt/venv
