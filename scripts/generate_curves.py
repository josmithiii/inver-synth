#!/usr/bin/env python3
"""
Generate training curves from CSV files
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def generate_curves(csv_file='output/InverSynth_e2e.csv', output_file='training_curves.png'):
    """Generate training curves plot"""
    
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 10:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Loss curves
                if 'loss' in df.columns:
                    ax1.plot(df['loss'], label='Training Loss')
                if 'val_loss' in df.columns:
                    ax1.plot(df['val_loss'], label='Validation Loss')
                ax1.set_title('Loss Curves')
                ax1.legend()
                ax1.grid(True)
                
                # Accuracy curves
                if 'top_k_mean_accuracy' in df.columns:
                    ax2.plot(df['top_k_mean_accuracy'], label='Training Accuracy')
                if 'val_top_k_mean_accuracy' in df.columns:
                    ax2.plot(df['val_top_k_mean_accuracy'], label='Validation Accuracy')
                ax2.set_title('Accuracy Curves')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                print('Training curves saved')
            else:
                print('CSV file is empty - creating placeholder plot')
                create_placeholder_plot(output_file, 'No training metrics available\n(CSV file is empty)')
        except Exception as e:
            print(f'Error reading CSV: {e} - creating placeholder plot')
            create_placeholder_plot(output_file, f'Error loading training metrics\n{e}')
    else:
        print('CSV file not found or empty - creating placeholder plot')
        create_placeholder_plot(output_file, 'No training metrics file found')
    
    print('✅ Training curves created')

def create_placeholder_plot(output_file, message):
    """Create a placeholder plot with a message"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
    ax.set_title('Training Curves - No Data')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'output/InverSynth_e2e.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'training_curves.png'
    generate_curves(csv_file, output_file)