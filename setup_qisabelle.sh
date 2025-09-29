#!/bin/bash
set -e

# hardcoded afp version from qisabelle default (2024 main release)
AFP_ID="2024_361b8b643a1d"

# quick check if everything is already setup
if [ -d "./qisabelle/dockerheaps/Isabelle2024_afp_$AFP_ID" ] && [ -d "./qisabelle/afp_$AFP_ID" ]; then
  echo "qisabelle already setup, skipping..."
  exit 0
fi

echo "setting up qisabelle with afp version: $AFP_ID"
echo ""

# check if we're in the right directory structure
if [ ! -f "pyproject.toml" ]; then
  echo "warning: no pyproject.toml found - make sure you're in your main project directory"
fi

# create qisabelle directory if it doesn't exist
if [ ! -d "./qisabelle" ]; then
  echo "cloning qisabelle repository..."
  git clone https://github.com/chrissiwaffler/qisabelle
else
  echo "qisabelle directory already exists, skipping clone"
fi

cd ./qisabelle/

# download afp release (theory files)
if [ ! -d "afp_$AFP_ID" ]; then
  echo "downloading afp theory files..."
  curl -u u363828-sub1:7K5XEQ7RDqvbjY8v \
    https://u363828-sub1.your-storagebox.de/afp_$AFP_ID.tar.gz -O

  echo "extracting afp theory files..."
  tar -xf afp_$AFP_ID.tar.gz
  rm afp_$AFP_ID.tar.gz
  echo "afp theory files ready"
else
  echo "afp theory files already exist, skipping download"
fi

# create dockerheaps directory
mkdir -p dockerheaps
cd dockerheaps

# download and decompress heaps
if [ ! -d "Isabelle2024_afp_$AFP_ID" ]; then
  echo "downloading isabelle heaps (this may take a while, ~7gb download)..."
  curl -u u363828-sub1:7K5XEQ7RDqvbjY8v \
    https://u363828-sub1.your-storagebox.de/Isabelle2024_afp_$AFP_ID.tar.br -O

  echo "decompressing heaps (requires brotli)..."
  tar --use-compress-program=brotli -xf Isabelle2024_afp_$AFP_ID.tar.br
  rm Isabelle2024_afp_$AFP_ID.tar.br
  echo "heaps ready (~40gb after decompression)"
else
  echo "isabelle heaps already exist, skipping download"
fi

cd .. # back to qisabelle directory

# fix permissions if needed
echo "setting permissions..."
chmod -R a+rwX afp_$AFP_ID/ || true

# build docker image with podman if needed
echo "building qisabelle server image..."
if ! podman image exists localhost/qisabelle-server:latest; then
  podman build -t qisabelle-server -f ServerDockerfile --format docker .
else
  echo "qisabelle server image already exists"
fi

cd .. # back to main project directory

echo ""
echo "qisabelle setup complete!"
echo ""
echo "directory structure:"
echo "├── qisabelle/"
echo "│   ├── afp_$AFP_ID/"
echo "│   ├── dockerheaps/"
echo "│   │   └── Isabelle2024_afp_$AFP_ID/"
echo "│   └── ..."
echo ""
echo "to start qisabelle server:"
echo "  cd qisabelle"
echo "  podman-compose --podman-build-args='--format docker' up"
echo ""
echo "to test the python client (in another terminal):"
echo "  cd qisabelle"
echo "  python -um client.main"
