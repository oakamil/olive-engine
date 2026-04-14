#!/bin/bash
set -e

# Create and activate a virtual environment if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Ensure maturin is installed
if ! command -v maturin &> /dev/null || ! command -v patchelf &> /dev/null; then
    echo "maturin/patchelf not found. Installing via pip..."
    pip install "maturin[patchelf]"
fi

echo "Cleaning up existing shared libraries..."
find target/release -maxdepth 2 -name "libolive_engine.so" -delete
rm -rf target/maturin

echo "Building Python wheel with Maturin..."
maturin build --release

echo "Build complete."
cp target/wheels/*manylinux*.whl dist/.

