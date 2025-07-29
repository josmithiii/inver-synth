# InverSynth Makefile
# Automates the complete neural network training and evaluation pipeline
# with proper dependency management

# Configuration
PYTHON := /Users/jos/miniforge3/envs/inver-synth/bin/python
DATASET_SIZE := 150
SAMPLE_RATE := 16384
AUDIO_LENGTH := 1.0

# Device detection - auto-detect best available device (CUDA > MPS > CPU)
DETECTED_DEVICE := $(shell $(PYTHON) detect_device.py 2>/dev/null || echo "cpu")
TF_DEVICE := $(or $(TF_DEVICE),$(DETECTED_DEVICE))

# File targets
DATASET_FILE := test_datasets/InverSynth_data.hdf5
PARAMS_FILE := test_datasets/InverSynth_params.pckl
MODEL_E2E := output/InverSynth_e2e.keras
MODEL_C1 := output/InverSynth_C1.keras
MODEL_C3 := output/InverSynth_C3.keras
MODEL_C6 := output/InverSynth_C6.keras
MODEL_C6XL := output/InverSynth_C6XL.keras

# Source files that affect dataset generation
GENERATOR_SOURCES := generators/fm_generator.py generators/generator.py generators/parameters.py generators/sine_generator.py generators/vst_generator.py

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
	@echo "Device Configuration:"
	@echo "  Current device: $(TF_DEVICE) (auto-detected: $(DETECTED_DEVICE))"
	@echo "  Override with: TF_DEVICE=cuda/mps/cpu make model-*"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Initialize project directories"
	@echo "  dataset        - Generate training dataset ($(DATASET_SIZE) examples, only if source files changed)"
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
	@echo "  clean          - Remove all generated files:"
	@echo "  clean-dataset  - Remove dataset only"
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

# Dataset generation - only regenerates when generator source files change
dataset: $(DATASET_FILE)

$(DATASET_FILE): $(GENERATOR_SOURCES) | test_datasets/ test_waves/ output/
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
	@echo "🔥 Training E2E CNN model on $(TF_DEVICE)..."
	TF_DEVICE=$(TF_DEVICE) $(PYTHON) -m models.e2e_cnn
	@echo "✅ E2E model trained: $(MODEL_E2E)"

model-c1: $(MODEL_C1)

$(MODEL_C1): $(DATASET_FILE)
	@echo "📊 Training C1 architecture (2 conv layers) on $(TF_DEVICE)..."
	TF_DEVICE=$(TF_DEVICE) $(PYTHON) -m models.spectrogram_cnn --model C1
	@echo "✅ C1 model trained: $(MODEL_C1)"

model-c3: $(MODEL_C3)

$(MODEL_C3): $(DATASET_FILE)
	@echo "📊 Training C3 architecture (4 conv layers) on $(TF_DEVICE)..."
	TF_DEVICE=$(TF_DEVICE) $(PYTHON) -m models.spectrogram_cnn --model C3
	@echo "✅ C3 model trained: $(MODEL_C3)"

model-c6: $(MODEL_C6)

$(MODEL_C6): $(DATASET_FILE)
	@echo "📊 Training C6 architecture (7 conv layers) on $(TF_DEVICE)..."
	TF_DEVICE=$(TF_DEVICE) $(PYTHON) -m models.spectrogram_cnn --model C6
	@echo "✅ C6 model trained: $(MODEL_C6)"

model-c6xl: $(MODEL_C6XL)

$(MODEL_C6XL): $(DATASET_FILE)
	@echo "📊 Training C6XL architecture (7 conv layers, XL) on $(TF_DEVICE)..."
	TF_DEVICE=$(TF_DEVICE) $(PYTHON) -m models.spectrogram_cnn --model C6XL
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
	$(PYTHON) -c "import matplotlib.pyplot as plt; import pandas as pd; \
		df = pd.read_csv('output/InverSynth_e2e.csv'); \
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)); \
		ax1.plot(df.get('loss', []), label='Training Loss'); \
		ax1.plot(df.get('val_loss', []), label='Validation Loss'); \
		ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True); \
		ax2.plot(df.get('top_k_mean_accuracy', []), label='Training Accuracy'); \
		ax2.plot(df.get('val_top_k_mean_accuracy', []), label='Validation Accuracy'); \
		ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(True); \
		plt.tight_layout(); plt.savefig('$(TRAINING_CURVES)'); plt.close()"
	@echo "✅ Training curves saved: $(TRAINING_CURVES)"

detailed-analysis: $(MODEL_E2E) $(PARAMS_FILE)
	@echo "🔍 Running detailed parameter analysis..."
	@$(PYTHON) -c " \
		from models.comparison import run_comparison; \
		from generators.fm_generator import InverSynthGenerator; \
		from tensorflow import keras; \
		from models.app import top_k_mean_accuracy; \
		import os; \
		generator = InverSynthGenerator(); \
		models = ['$(MODEL_E2E)', '$(MODEL_C1)', '$(MODEL_C3)', '$(MODEL_C6)', '$(MODEL_C6XL)']; \
		for model_file in models: \
			if os.path.exists(model_file): \
				print(f'📈 Analyzing {model_file}...'); \
				model = keras.models.load_model(model_file, custom_objects={'top_k_mean_accuracy': top_k_mean_accuracy}); \
				model_name = os.path.basename(model_file).replace('.keras', '').replace('.h5', ''); \
				run_comparison(model, generator, 'InverSynth', num_samples=3, output_dir=f'detailed_comparison_{model_name}') \
	"
	@echo "✅ Detailed analysis complete!"

# Testing
test: $(DATASET_FILE)
	@echo "🧪 Running test suite..."
	$(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests passed!"

# Reconstruction utilities
reconstruct-sample: $(MODEL_E2E) $(PARAMS_FILE)
	@echo "🎼 Reconstructing sample audio..."
	$(PYTHON) -c " \
		from reconstruction.fm_reconstruction import FMResynth; \
		import os; \
		resynth = FMResynth(); \
		sample_file = 'test_waves/InverSynth/InverSynth_00042.wav'; \
		if os.path.exists(sample_file): \
			resynth.reconstruct('$(MODEL_E2E)', '$(PARAMS_FILE)', sample_file, 'sample_reconstruction.wav'); \
			print('✅ Sample reconstruction saved: sample_reconstruction.wav') \
		else: \
			print('❌ Sample file not found. Generate dataset first.') \
	"

# Cleaning targets
clean: clean-models clean-results clean-dataset
	@echo "✅ All generated files removed!"

clean-models:
	@echo "🗑️  Removing trained models..."
	@rm -f output/*.h5 output/*.keras output/*.csv

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
