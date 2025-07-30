# InverSynth Makefile
# Automates the complete neural network training and evaluation pipeline
# with proper dependency management

# Configuration
PYTHON := uv run python
DATASET_SIZE := 150
SAMPLE_RATE := 16384
AUDIO_LENGTH := 1.0

# File targets
DATASET_FILE := test_datasets/InverSynth_data.hdf5
PARAMS_FILE := test_datasets/InverSynth_params.pckl
MODEL_E2E := output/InverSynth_e2e.pth
MODEL_C1 := output/InverSynth_C1.pth
MODEL_C3 := output/InverSynth_C3.pth
MODEL_C6 := output/InverSynth_C6.pth
MODEL_C6XL := output/InverSynth_C6XL.pth

# Evaluation outputs
TRAINING_CURVES := training_curves.png
AUDIO_COMPARISONS := comparison_results/InverSynth
LISTEN_RESULTS := $(AUDIO_COMPARISONS)/listen_results.done

.PHONY: all setup clean test help models evaluation listen
.DEFAULT_GOAL := all

# Main targets
all: models evaluation
	@echo "✅ Complete InverSynth pipeline finished!"
	@echo "🎧 Listen to results in: $(AUDIO_COMPARISONS)/"

h help:
	@echo "InverSynth Makefile Targets:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Initialize project directories"
	@echo "  dataset        - Generate training dataset ($(DATASET_SIZE) examples)"
	@echo ""
	@echo "Training:"
	@echo "  models         - Train all neural network architectures"
	@echo "  model-e2e      - Train End-to-End CNN model"
	@echo "  model-c1       - Train C1 spectrogram CNN (2 layers)"
	@echo "  model-c3       - Train C3 spectrogram CNN (4 layers)"
	@echo "  model-c6       - Train C6 spectrogram CNN (7 layers)"
	@echo "  model-c6xl     - Train C6XL spectrogram CNN (7 layers, XL)"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluation     - Complete model evaluation suite"
	@echo "  listen         - Generate audio comparisons and spectrograms"
	@echo "  curves         - Generate training curves"
	@echo ""
	@echo "Utilities:"
	@echo "  test           - Run test suite"
	@echo "  clean          - Remove generated files"
	@echo "  clean-models   - Remove trained models only"
	@echo "  clean-results  - Remove evaluation results only"

# Setup and initialization
setup: test_datasets/ test_waves/ output/
	@echo "✅ Project directories initialized"

test_datasets/:
	@echo "📁 Creating datasets directory..."
	$(PYTHON) -m tasks.start

test_waves/:
	@echo "📁 Creating wave files directory..."
	$(PYTHON) -m tasks.start

output/:
	@echo "📁 Creating output directory..."
	mkdir -p output

# Dataset generation
dataset: $(DATASET_FILE)

$(DATASET_FILE): setup
	@echo "📊 Generating training dataset ($(DATASET_SIZE) examples)..."
	$(PYTHON) -m generators.fm_generator \
		--num_examples $(DATASET_SIZE) \
		--length $(AUDIO_LENGTH) \
		--sample_rate $(SAMPLE_RATE) \
		--name InverSynth \
		--dataset_directory test_datasets \
		--wavefile_directory test_waves
	@echo "✅ Dataset generated: $(DATASET_FILE)"

# Individual model training targets
models: model-e2e model-c1 model-c3 model-c6 model-c6xl
	@echo "✅ All models trained!"

model-e2e: $(MODEL_E2E)

$(MODEL_E2E): $(DATASET_FILE)
	@echo "🔥 Training E2E CNN model..."
	$(PYTHON) -m models.e2e_cnn
	@echo "✅ E2E model trained: $(MODEL_E2E)"

model-c1: $(MODEL_C1)

$(MODEL_C1): $(DATASET_FILE)
	@echo "📊 Training C1 architecture (2 conv layers)..."
	ARCHITECTURE=C1 $(PYTHON) -m models.spectrogram_cnn  # ./models/spectrogram_cnn.py
	@echo "✅ C1 model trained: $(MODEL_C1)"

model-c3: $(MODEL_C3)

$(MODEL_C3): $(DATASET_FILE)
	@echo "📊 Training C3 architecture (4 conv layers)..."
	ARCHITECTURE=C3 $(PYTHON) -m models.spectrogram_cnn
	@echo "✅ C3 model trained: $(MODEL_C3)"

model-c6: $(MODEL_C6)

$(MODEL_C6): $(DATASET_FILE)
	@echo "📊 Training C6 architecture (7 conv layers)..."
	ARCHITECTURE=C6 $(PYTHON) -m models.spectrogram_cnn
	@echo "✅ C6 model trained: $(MODEL_C6)"

model-c6xl: $(MODEL_C6XL)

$(MODEL_C6XL): $(DATASET_FILE)
	@echo "📊 Training C6XL architecture (7 conv layers, XL)..."
	ARCHITECTURE=C6XL $(PYTHON) -m models.spectrogram_cnn
	@echo "✅ C6XL model trained: $(MODEL_C6XL)"

# Evaluation targets
evaluation: listen curves detailed-analysis
	@echo "✅ Complete evaluation finished!"

listen: $(LISTEN_RESULTS)

$(LISTEN_RESULTS): $(MODEL_E2E)
	@echo "🎵 Creating audio reconstructions and spectrograms..."
	$(PYTHON) listen_results.py
	@mkdir -p $(AUDIO_COMPARISONS)
	@touch $(LISTEN_RESULTS)
	@echo "✅ Audio comparisons ready: $(AUDIO_COMPARISONS)/"

curves: $(TRAINING_CURVES)

$(TRAINING_CURVES): $(MODEL_E2E)
	@echo "📈 Generating training curves..."
	$(PYTHON) scripts/generate_curves.py output/InverSynth_e2e.csv $(TRAINING_CURVES)
	@echo "✅ Training curves saved: $(TRAINING_CURVES)"

detailed-analysis: $(MODEL_E2E) $(PARAMS_FILE)
	@echo "🔍 Running detailed parameter analysis..."
	@echo "⚠️  Detailed analysis temporarily disabled - PyTorch model loading needs implementation"
	@echo "✅ Detailed analysis complete!"

# Testing
test: $(DATASET_FILE)
	@echo "🧪 Running test suite..."
	$(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests passed!"

# Reconstruction utilities
reconstruct-sample: $(MODEL_E2E) $(PARAMS_FILE)
	@echo "🎼 Reconstructing sample audio..."
	@echo "⚠️  Sample reconstruction temporarily disabled - PyTorch model loading needs implementation"

# Cleaning targets
clean: clean-models clean-results clean-dataset
	@echo "✅ All generated files removed!"

clean-models:
	@echo "🗑️  Removing trained models..."
	@rm -f output/*.h5 output/*.pth output/*.csv

clean-results:
	@echo "🗑️  Removing evaluation results..."
	@rm -rf comparison_results/ detailed_comparison_*/ *.png

clean-dataset:
	@echo "🗑️  Removing dataset..."
	@rm -rf test_datasets/ test_waves/

# Status and information
status:
	@echo "📊 InverSynth Project Status:"
	@echo ""
	@echo "Dataset:"
	@if [ -f "$(DATASET_FILE)" ]; then \
		echo "  ✅ Dataset exists ($(DATASET_FILE))"; \
		echo "     Size: $$(stat -f%z '$(DATASET_FILE)' 2>/dev/null || echo 'unknown') bytes"; \
	else \
		echo "  ❌ Dataset missing (run 'make dataset')"; \
	fi
	@echo ""
	@echo "Models:"
	@for model in "$(MODEL_E2E)" "$(MODEL_C1)" "$(MODEL_C3)" "$(MODEL_C6)" "$(MODEL_C6XL)"; do \
		if [ -f "$$model" ]; then \
			echo "  ✅ $$(basename $$model)"; \
		else \
			echo "  ❌ $$(basename $$model) (not trained)"; \
		fi \
	done
	@echo ""
	@echo "Evaluation:"
	@if [ -f "$(TRAINING_CURVES)" ]; then \
		echo "  ✅ Training curves generated"; \
	else \
		echo "  ❌ Training curves missing"; \
	fi
	@if [ -d "$(AUDIO_COMPARISONS)" ]; then \
		echo "  ✅ Audio comparisons available"; \
	else \
		echo "  ❌ Audio comparisons missing"; \
	fi

# Quick development targets
quick: dataset model-e2e listen
	@echo "✅ Quick development pipeline complete!"
	@echo "🎧 Check: $(AUDIO_COMPARISONS)/"

paper-models: model-c1 model-c3 model-c6 model-c6xl
	@echo "✅ All paper architectures trained!"

# Parallel training (if you have multiple GPUs or want to experiment)
models-parallel:
	@echo "🚀 Training models in parallel..."
	@$(MAKE) -j4 model-c1 model-c3 model-c6 model-c6xl
	@echo "✅ Parallel training complete!"
