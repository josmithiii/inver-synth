#!/usr/bin/env python3
"""
Listen to InverSynth model results - compare original vs reconstructed audio
Also generates spectrograms for visual comparison
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from models.app import top_k_mean_accuracy
from models.comparison import run_comparison
from generators.fm_generator import InverSynthGenerator

def plot_spectrogram_comparison(original_file, reconstructed_file, output_dir):
    """Plot spectrograms of original vs reconstructed audio side by side"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Load audio files
    y_orig, sr_orig = librosa.load(original_file, sr=16384)
    y_recon, sr_recon = librosa.load(reconstructed_file, sr=16384)
    
    # Compute spectrograms
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    S_recon = librosa.amplitude_to_db(np.abs(librosa.stft(y_recon)), ref=np.max)
    
    # Plot original
    librosa.display.specshow(S_orig, x_axis='time', y_axis='hz', sr=sr_orig, ax=axes[0])
    axes[0].set_title('Original Audio Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')
    
    # Plot reconstructed
    librosa.display.specshow(S_recon, x_axis='time', y_axis='hz', sr=sr_recon, ax=axes[1])
    axes[1].set_title('Reconstructed Audio Spectrogram')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save plot
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    plot_file = f"{output_dir}/{base_name}_spectrograms.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Spectrogram comparison saved: {plot_file}")
    plt.close()

def create_comparison_samples(model_file, num_samples=5):
    """Create audio comparisons using the trained model"""
    
    print(f"🎵 Creating audio comparisons with model: {model_file}")
    
    # Load model - try PyTorch first, fall back to Keras
    if model_file.endswith('.pth'):
        # PyTorch model
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Determine model architecture from file name
        if 'e2e' in model_file:
            from models.e2e_cnn import E2EModel
            from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers
            model = E2EModel(
                n_outputs=256,
                c1d_layers=cE2E_1d_layers,
                c2d_layers=cE2E_2d_layers,
                input_size=16384
            )
        else:
            # Spectrogram model - extract architecture from filename
            from models.spectrogram_cnn import SpectrogramModel
            from models.common.architectures import get_architecture_layers
            
            # Extract architecture from filename (e.g., "InverSynth_C6.pth" -> "C6")
            arch_name = 'C1'  # Default
            for arch in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6XL']:
                if arch in model_file:
                    arch_name = arch
                    break
            
            layers = get_architecture_layers(arch_name)
            model = SpectrogramModel(n_outputs=256, layers=layers)
        
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        # Legacy Keras model - convert filename to PyTorch equivalent
        pth_file = model_file.replace('.h5', '.pth')
        if os.path.exists(pth_file):
            # Use PyTorch version if available
            checkpoint = torch.load(pth_file, map_location='cpu')
            
            if 'e2e' in pth_file:
                from models.e2e_cnn import E2EModel
                from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers
                model = E2EModel(
                    n_outputs=256,
                    c1d_layers=cE2E_1d_layers,
                    c2d_layers=cE2E_2d_layers,
                    input_size=16384
                )
            else:
                from models.spectrogram_cnn import SpectrogramModel
                from models.common.architectures import get_architecture_layers
                
                # Extract architecture from filename
                arch_name = 'C1'  # Default
                for arch in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6XL']:
                    if arch in pth_file:
                        arch_name = arch
                        break
                
                layers = get_architecture_layers(arch_name)
                model = SpectrogramModel(n_outputs=256, layers=layers)
                
            model.load_state_dict(checkpoint)
            model.eval()
        else:
            raise FileNotFoundError(f"Neither PyTorch ({pth_file}) nor Keras ({model_file}) model found")
    
    # Set up FM generator
    generator = InverSynthGenerator()
    
    # Create comparison directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparison with random samples
    run_comparison(
        model=model,
        generator=generator,
        run_name="InverSynth",
        num_samples=num_samples,
        data_dir="test_datasets",
        output_dir=output_dir,
        length=1.0,
        sample_rate=16384,
        shuffle=True
    )
    
    # Generate spectrograms for each comparison
    comparison_dir = f"{output_dir}/InverSynth"
    if os.path.exists(comparison_dir):
        for file in os.listdir(comparison_dir):
            if file.endswith("_copy.wav"):
                base_name = file.replace("_copy.wav", "")
                original_file = f"{comparison_dir}/{base_name}_copy.wav"
                reconstructed_file = f"{comparison_dir}/{base_name}_reconstruct.wav"
                
                if os.path.exists(reconstructed_file):
                    plot_spectrogram_comparison(original_file, reconstructed_file, comparison_dir)
    
    print(f"\n✅ Comparison complete! Results in: {comparison_dir}/")
    print("\n🎧 To listen to results:")
    print(f"   Original:      {comparison_dir}/*_copy.wav")
    print(f"   Reconstructed: {comparison_dir}/*_reconstruct.wav")
    print(f"   Duplicate:     {comparison_dir}/*_duplicate.wav")
    print("\n📊 Spectrograms: {comparison_dir}/*_spectrograms.png")

def analyze_training_metrics():
    """Analyze the training metrics from CSV file"""
    import pandas as pd
    
    csv_file = "output/InverSynth_e2e.csv"
    if os.path.exists(csv_file):
        print("📈 Training metrics analysis:")
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print("   No training metrics found (CSV file is empty)")
                return
        except pd.errors.EmptyDataError:
            print("   No training metrics found (CSV file is empty)")
            return
        
        # Print summary statistics
        print(f"   Training epochs: {len(df)}")
        if 'loss' in df.columns:
            print(f"   Final training loss: {df['loss'].iloc[-1]:.4f}")
        if 'val_loss' in df.columns:
            print(f"   Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
        if 'top_k_mean_accuracy' in df.columns:
            print(f"   Final top-k accuracy: {df['top_k_mean_accuracy'].iloc[-1]:.4f}")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss curves
        if 'loss' in df.columns:
            axes[0].plot(df['loss'], label='Training Loss')
        if 'val_loss' in df.columns:
            axes[0].plot(df['val_loss'], label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        if 'top_k_mean_accuracy' in df.columns:
            axes[1].plot(df['top_k_mean_accuracy'], label='Training Accuracy')
        if 'val_top_k_mean_accuracy' in df.columns:
            axes[1].plot(df['val_top_k_mean_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Top-K Mean Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
        print("📊 Training curves saved: training_curves.png")
        plt.close()

if __name__ == "__main__":
    print("🎼 InverSynth Model Results Analysis")
    print("="*50)
    
    # Check available models - prefer PyTorch over Keras
    models = []
    
    # Check for PyTorch models first
    pytorch_models = [
        "output/InverSynth_e2e.pth",
        "output/InverSynth_C1.pth", 
        "output/InverSynth_C3.pth",
        "output/InverSynth_C6.pth",
        "output/InverSynth_C6XL.pth"
    ]
    
    for model_file in pytorch_models:
        if os.path.exists(model_file):
            models.append(model_file)
    
    # If no PyTorch models, check for Keras models
    if not models:
        keras_models = [
            "output/InverSynth_e2e.h5",
            "output/InverSynth_C1.h5",
            "output/InverSynth_C3.h5", 
            "output/InverSynth_C6.h5",
            "output/InverSynth_C6XL.h5"
        ]
        
        for model_file in keras_models:
            if os.path.exists(model_file):
                models.append(model_file)
    
    if not models:
        print("❌ No trained models found in output/ directory")
        print("Expected files: output/InverSynth_*.pth or output/InverSynth_*.h5")
        exit(1)
    
    print(f"📁 Found models: {models}")
    
    # Analyze training metrics
    analyze_training_metrics()
    
    # Create audio comparisons
    for model_file in models:
        create_comparison_samples(model_file, num_samples=3)
    
    print("\n🎯 Summary:")
    print("   1. Check training_curves.png for learning progress")  
    print("   2. Listen to audio files in comparison_results/InverSynth/")
    print("   3. Compare spectrograms in *_spectrograms.png files")
    print("   4. Original = copy, Reconstructed = reconstruct, Reference = duplicate")