# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the InverSynth method from the paper "Barkan, Tsiris, Koenigstein, Katz" (arXiv:1812.06349). The project implements neural networks that can predict synthesizer parameters from audio input, supporting both End-to-End (E2E) and STFT spectrogram-based approaches.

## Commands

### Environment Setup
```bash
uv sync --extra dev
```

### Main Operations
- `uv run task start` - Initialize project (creates directories and .env file)
- `uv run task generate` - Generate training dataset with default parameters
- `uv run task test` - Run test suite
- `uv run task clean` - Run linting, formatting, and type checking

### Manual Dataset Generation
```bash
uv run python -m generators.fm_generator [options]
```

### Model Training
```bash
uv run python -m models.e2e_cnn        # End-to-end CNN model
uv run python -m models.spectrogram_cnn # STFT spectrogram CNN model
```

### Development Tools
- Linting: `uv run flake8 models/ generators/ tests/`
- Formatting: `uv run black models/ generators/ tests/`
- Import sorting: `uv run isort .`
- Type checking: `uv run mypy` (configured in mypy.ini)

## Architecture

### Core Components

**Generators** (`generators/`):
- `SoundGenerator` base class for audio synthesis
- `DatasetCreator` handles parameter generation and audio file creation
- Multiple generator implementations (FM, sine, VST plugins)
- Parameter system with encoding/decoding for neural network training

**Models** (`models/`):
- Two main approaches: E2E CNNs and Spectrogram CNNs
- Multiple architecture variants (C1-C6, C6XL, CE2E) defined in `common/architectures.py`
- Shared training pipeline in `app.py` with configurable model callbacks
- Custom metrics: top-k mean accuracy, mean absolute error

**Data Pipeline**:
- HDF5-based dataset storage with metadata
- Audio normalization and preprocessing
- Train/validation splitting with generators
- Support for channels_first/channels_last data formats

### Key Architecture Files
- `models/app.py` - Central training orchestration and evaluation
- `models/common/architectures.py` - CNN layer definitions for all model variants
- `generators/generator.py` - Base classes for dataset creation and audio generation
- `generators/parameters.py` - Parameter encoding/decoding system

### Model Training Flow
1. Dataset creation: parameter sampling → audio generation → HDF5 storage
2. Model selection: Choose architecture (C1-C6XL, E2E variants)
3. Training: Uses data generators, early stopping, model checkpointing
4. Evaluation: Custom metrics comparing predicted vs actual synthesizer parameters

### Configuration
- Environment variables in `.env` for model/training configuration
- uv-based dependency management with locked versions
- Taskipy for common development tasks
- Plugin configurations in `plugin_config/` for VST-based generation

## Development Notes

- Uses TensorFlow 2.1.0 with Keras for neural networks
- Audio processing with librosa and custom spectrogram handling via kapre
- Type hints expected throughout codebase
- Tests located in `tests/` directory using pytest framework
- Supports both CPU and GPU training (CUDA detection included)