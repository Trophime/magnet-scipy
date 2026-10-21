# Build the magnet-scipy runtime images, one per matplotlib backend.
#
#   VERSION=$(python3 -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") \
#       docker buildx bake                      # all three images
#   docker buildx bake magnet-scipy-qtagg       # a single one
#
# See docker/README.md.

variable "VERSION" {
  default = "dev"
}

group "default" {
  targets = ["magnet-scipy-tkagg", "magnet-scipy-qtagg", "magnet-scipy-webagg"]
}

target "magnet-scipy" {
  name = "magnet-scipy-${item.backend}"
  matrix = {
    item = [
      { backend = "tkagg", extras = "" },
      { backend = "qtagg", extras = "--extra gui" },
      { backend = "webagg", extras = "--extra web" },
    ]
  }
  context    = "."
  dockerfile = "docker/Dockerfile"
  target     = item.backend
  args = {
    UV_EXTRAS = item.extras
  }
  tags = [
    "magnet-scipy:${VERSION}-${item.backend}",
    "magnet-scipy:${item.backend}",
  ]
  labels = {
    "org.opencontainers.image.version" = VERSION
  }
}
