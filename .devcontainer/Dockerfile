FROM debian:trixie

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Debian trixie ships Python 3.13; build-essential/make cover the Sphinx Makefile.
# python3-tk provides the TkAgg matplotlib backend (X11, no OpenGL needed).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        make \
        python3 \
        python3-dev \
        python3-tk \
        python3-venv \
        sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Use the system Python and keep the venv outside the bind-mounted workspace,
# so it does not clash with a host-side .venv.
# MPLBACKEND is left unset: matplotlib picks TkAgg when DISPLAY is set, Agg otherwise.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/$USERNAME/.venv \
    PATH=/home/$USERNAME/.venv/bin:$PATH
