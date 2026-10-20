#!/bin/sh
# Runs on the host before each container start (initializeCommand).
# Exports the current X11 cookie as a FamilyWild entry, so it is valid whatever
# the container hostname, into a fixed directory bind-mounted into the container.
# Needed because the host cookie file (e.g. GNOME's .mutter-Xwaylandauth.*)
# is renamed at every login.
umask 077
dir="$HOME/.cache/magnet-scipy-devcontainer"
mkdir -p "$dir"
: > "$dir/Xauthority"
if [ -n "$DISPLAY" ] && command -v xauth >/dev/null; then
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$dir/Xauthority" nmerge -
fi
exit 0
