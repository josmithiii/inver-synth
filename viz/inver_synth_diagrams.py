#!/usr/bin/env python3
"""InverSynth model diagram generation."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.common.architectures import layers_map, get_architecture_layers
from models.spectrogram_cnn import SpectrogramCNN
from models.e2e_cnn import E2EModel


def create_text_summary(model, input_shape=(1, 1, 16384), model_name="Model"):
    """Create a text summary of the model architecture."""
    print("=" * 80)
    print(f"{model_name} Architecture Summary")
    print("=" * 80)

    # Print model structure
    print("\nModel Structure:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nParameter Count:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass and show shapes
    print("\nForward Pass Shape Analysis:")
    print(f"Input shape: {input_shape}")

    with torch.no_grad():
        x = torch.randn(*input_shape)
        print(f"Input: {x.shape}")

        try:
            # Get intermediate shapes by hooking into the model
            if hasattr(model, 'spectrogram'):
                # Spectrogram CNN model
                spec_out = model.spectrogram(x.squeeze(1))
                db_out = model.amplitude_to_db(spec_out)
                db_out = db_out.unsqueeze(1)
                print(f"After spectrogram: {spec_out.shape}")
                print(f"After amplitude_to_db + unsqueeze: {db_out.shape}")
                
                # Permute dimensions
                x_perm = db_out.permute(0, 1, 3, 2)
                print(f"After permute (batch, channel, time, freq): {x_perm.shape}")
                
                # Apply conv layers
                conv_x = x_perm
                for i, conv_layer in enumerate(model.conv2d_layers):
                    conv_x = torch.nn.functional.relu(conv_layer(conv_x))
                    print(f"After conv2d layer {i+1}: {conv_x.shape}")
                
                # Flatten
                flat_x = conv_x.view(conv_x.size(0), -1)
                print(f"After flatten: {flat_x.shape}")
                
            # Final output
            final_out = model(x)
            print(f"Final output: {final_out.shape}")

        except Exception as e:
            print(f"Error during forward pass analysis: {e}")
            final_out = model(x)
            print(f"Final output: {final_out.shape}")


def create_architecture_diagram(arch_name: str):
    """Create an ASCII diagram for a specific architecture."""
    print(f"\n{'=' * 80}")
    print(f"{arch_name} Architecture Flow Diagram")
    print("=" * 80)

    if arch_name not in layers_map:
        print(f"Unknown architecture: {arch_name}")
        return

    layers = layers_map[arch_name]
    
    if arch_name.startswith('C'):  # Spectrogram CNN architectures
        print(f"""
    Audio Input (16384 samples @ 16kHz)
           │
           ▼
    ┌─────────────────────────────┐
    │    STFT PREPROCESSING       │
    │  Spectrogram(n_fft=512,     │  ← Convert to frequency domain
    │              hop=256)       │  ← Power=2.0 for power spec
    │  AmplitudeToDB()            │  ← Convert to dB scale
    │  Permute(time, freq)        │  ← Rearrange dimensions
    └─────────────────────────────┘
           │ Shape: (batch, 1, 65, 257)
           ▼""")
        
        # Show each conv layer
        current_shape = "(batch, 1, 65, 257)"
        in_channels = 1
        
        for i, layer in enumerate(layers):
            # Handle both tuple and int window_size
            if isinstance(layer.window_size, tuple):
                kernel_str = f"{layer.window_size[0]}×{layer.window_size[1]}"
            else:
                kernel_str = f"{layer.window_size}×{layer.window_size}"
                
            print(f"""    ┌─────────────────────────────┐
    │     CONV BLOCK {i+1:2d}           │
    │  Conv2d({in_channels:3d}→{layer.filters:3d}, {kernel_str},  │  ← {kernel_str} convolution
    │          stride={layer.strides}, pad=auto) │  ← stride {layer.strides}
    │  ReLU()                     │  ← activation function
    └─────────────────────────────┘""")
            in_channels = layer.filters
        
        print(f"""           │ Final conv shape
           ▼
    ┌─────────────────────────────┐
    │    FULLY CONNECTED          │
    │  Flatten()                  │  ← reshape to 1D
    │  Linear(conv_out → 512)     │  ← hidden layer
    │  ReLU()                     │  ← activation
    │  Linear(512 → n_outputs)    │  ← output layer
    │  Sigmoid()                  │  ← final activation
    └─────────────────────────────┘
           │
           ▼
        Synthesizer Parameters
        """)
    
    elif arch_name.startswith('CE2E'):  # End-to-end architectures
        # For CE2E, we need to get both 1D and 2D layers
        from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers
        
        if arch_name == 'CE2E':
            layers_1d = cE2E_1d_layers
            layers_2d = []
        else:  # CE2E_2D
            layers_1d = cE2E_1d_layers
            layers_2d = cE2E_2d_layers
            
        print(f"""
    Raw Audio Input (16384 samples @ 16kHz)
           │ Direct audio processing - no STFT
           ▼""")
        
        # Show 1D conv layers
        for i, layer in enumerate(layers_1d):
            kernel_size = layer.window_size if isinstance(layer.window_size, int) else layer.window_size[0]
            stride = layer.strides if isinstance(layer.strides, int) else layer.strides[0]
            print(f"""    ┌─────────────────────────────┐
    │     1D CONV BLOCK {i+1}         │
    │  Conv1d(filters={layer.filters:3d},      │  ← 1D convolution
    │         kernel={kernel_size},           │  ← kernel size {kernel_size}
    │         stride={stride})               │  ← stride {stride}
    │  ReLU()                     │  ← activation function
    └─────────────────────────────┘
           │
           ▼""")
        
        if layers_2d:
            print("""    ┌─────────────────────────────┐
    │      RESHAPE TO 2D          │
    │  Reshape for 2D processing  │
    └─────────────────────────────┘
           │
           ▼""")
            
            # Show 2D conv layers
            for i, layer in enumerate(layers_2d):
                kernel_str = f"{layer.window_size[0]}×{layer.window_size[1]}"
                print(f"""    ┌─────────────────────────────┐
    │     2D CONV BLOCK {i+1}         │
    │  Conv2d(filters={layer.filters:3d},      │  ← 2D convolution
    │         kernel={kernel_str},           │  ← kernel size {kernel_str}
    │         stride={layer.strides})        │  ← stride {layer.strides}
    │  ReLU()                     │  ← activation function
    └─────────────────────────────┘
           │
           ▼""")
        
        print("""    ┌─────────────────────────────┐
    │    FULLY CONNECTED          │
    │  Flatten()                  │  ← reshape to 1D
    │  Linear(conv_out → 512)     │  ← hidden layer
    │  ReLU()                     │  ← activation
    │  Linear(512 → n_outputs)    │  ← output layer
    │  Sigmoid()                  │  ← final activation
    └─────────────────────────────┘
           │
           ▼
        Synthesizer Parameters
        """)


def generate_model_diagram(arch_name: str, n_outputs: int = 12):
    """Generate diagram for a specific architecture."""
    print(f"\nGenerating diagram for architecture: {arch_name}")
    
    try:
        if arch_name.startswith('C') and arch_name != 'CE2E':
            # Spectrogram CNN
            layers = get_architecture_layers(arch_name)
            model = SpectrogramCNN(
                n_outputs=n_outputs,
                arch_layers=layers, 
                input_size=16384
            )
            input_shape = (1, 1, 16384)
            
        elif arch_name in ['CE2E', 'CE2E_2D']:
            # E2E CNN - need to check if this model exists
            print(f"E2E model visualization not yet implemented for {arch_name}")
            create_architecture_diagram(arch_name)
            return
            
        else:
            print(f"Unknown architecture: {arch_name}")
            return
            
        # Generate text summary
        create_text_summary(model, input_shape=input_shape, model_name=f"{arch_name} CNN")
        
        # Generate ASCII diagram
        create_architecture_diagram(arch_name)
        
    except Exception as e:
        print(f"Error generating diagram for {arch_name}: {e}")
        print("Generating architecture flow diagram only...")
        create_architecture_diagram(arch_name)


def list_architectures():
    """List all available architectures."""
    print("Available InverSynth Architectures:")
    print("=" * 40)
    
    spectrogram_archs = [k for k in layers_map.keys() if k.startswith('C') and k != 'CE2E']
    e2e_archs = [k for k in layers_map.keys() if k.startswith('CE2E')]
    
    print("\nSpectrogram CNN Architectures:")
    for arch in spectrogram_archs:
        layers = layers_map[arch]
        print(f"  {arch:8s} - {len(layers)} conv layers")
    
    print("\nEnd-to-End Architectures:")
    for arch in e2e_archs:
        layers = layers_map[arch]
        print(f"  {arch:8s} - {len(layers)} conv layers")


def main():
    parser = argparse.ArgumentParser(description="Generate InverSynth model architecture diagrams")
    parser.add_argument(
        "--arch", "-a", 
        default="C6", 
        help="Architecture name (default: C6)"
    )
    parser.add_argument(
        "--outputs", "-o",
        type=int,
        default=12,
        help="Number of output parameters (default: 12)"
    )
    parser.add_argument(
        "--list", "-l", 
        action="store_true", 
        help="List available architectures"
    )

    args = parser.parse_args()

    if args.list:
        list_architectures()
        return

    generate_model_diagram(args.arch, args.outputs)

    print(f"\n{'='*80}")
    print("Model diagram generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()