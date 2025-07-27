#!/bin/bash -e
# InverSynth Setup Script - Simple conda-only approach
# Run this on a new machine to set up the environment from scratch

set -o pipefail

echo "🔧 Setting up InverSynth environment..."

# Step 1: Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "❌ Error: Neither conda nor mamba found. Please install miniforge or miniconda."
    exit 1
fi

# Step 2: Fix deprecated pyproject.toml format if needed
echo "🔍 Checking pyproject.toml format..."
if grep -q "\[tool.poetry.dev-dependencies\]" pyproject.toml; then
    echo "🔧 Updating deprecated pyproject.toml format..."
    sed -i.bak 's/\[tool.poetry.dev-dependencies\]/[tool.poetry.group.dev.dependencies]/' pyproject.toml
    echo "✅ Updated pyproject.toml format"
fi

# Step 3: Create conda environment with ALL dependencies
ENV_NAME="inver-synth"
echo "🐍 Creating conda environment '$ENV_NAME'..."

# Remove existing environment if it exists
$CONDA_CMD env remove -n $ENV_NAME -y 2>/dev/null || true

# Create environment from requirements.txt + conda packages
echo "📦 Installing dependencies..."
$CONDA_CMD create -n $ENV_NAME python=3.10 -y

# Activate environment and install packages
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install scientific computing packages via conda
$CONDA_CMD install -y llvm llvmlite numba numpy scipy pandas matplotlib
# Use pip for tensorflow since 2.1.0 is too old for conda
pip install tensorflow

# Install remaining packages via pip
pip install -r requirements.txt

# Install development dependencies
pip install black isort mypy flake8 autoflake pytest taskipy

# Step 4: Initialize project
echo "📁 Initializing project directories..."
python -m tasks.start

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Usage:"
echo "  conda activate $ENV_NAME"
echo "  python -m tasks.start              # Initialize project"
echo "  python -m pytest                   # Run tests"
echo "  python -m generators.fm_generator  # Generate dataset"
echo "  python -m models.e2e_cnn           # Train E2E model" 
echo "  python -m models.spectrogram_cnn   # Train spectrogram model"
echo "  doit                               # All of the above"
echo ""
echo "📊 Environment info:"
echo "  Environment: $ENV_NAME"
echo "  Python: $(python --version)"
echo "  Location: $(conda info --envs | grep $ENV_NAME)"
echo ""
echo "🔍 To verify setup:"
echo "  python -c 'import tensorflow; import numba; print(\"✅ All dependencies working\")'"

