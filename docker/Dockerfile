# Runtime image for magnet-scipy.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t magnet-scipy .
# Run (current directory mounted as /data):
#   docker run --rm -v "$PWD":/data magnet-scipy magnet-scipy --help
# Run with plot windows (X11 forwarding, TkAgg backend):
#   docker run --rm --user "$(id -u):$(id -g)" \
#       -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -e XAUTHORITY=/tmp/.Xauthority -v "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
#       -v "$PWD":/data magnet-scipy magnet-scipy-coupled --show_plots ...

FROM debian:trixie AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Install locked dependencies first so this layer is cached across code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY magnet_scipy ./magnet_scipy
RUN uv sync --frozen --no-dev --no-editable


FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive

# python3-tk provides the TkAgg matplotlib backend (X11, no OpenGL needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# MPLBACKEND is left unset: matplotlib picks TkAgg when DISPLAY is set, Agg otherwise
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /data

CMD ["magnet-scipy", "--help"]
