import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.architectures import cE2E_1d_layers, cE2E_2d_layers

"""
End-to-End learning. A CNN predicts the synthesizer
parameter configuration directly from the raw audio.
The first convolutional layers perform 1D convolutions
that learn an alternative representation for the STFT
Spectrogram. Then, a stack of 2D convolutional layers
analyze the learned representation to predict the
synthesizer parameter configuration.
"""


"""Model Architecture"""
# @ paper:
# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).


class E2EModel(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        c1d_layers: list,
        c2d_layers: list,
        input_size: int,
        n_dft: int = 128,
        n_hop: int = 64,
    ):
        super(E2EModel, self).__init__()
        
        # 1D Convolutional layers
        self.conv1d_layers = nn.ModuleList()
        in_channels = 1  # raw audio has 1 channel
        
        for arch_layer in c1d_layers:
            self.conv1d_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=arch_layer.filters,
                    kernel_size=arch_layer.window_size,
                    stride=arch_layer.strides,
                    padding=0,
                )
            )
            in_channels = arch_layer.filters
        
        # 2D Convolutional layers
        self.conv2d_layers = nn.ModuleList()
        in_channels = 1  # after reshape, we have 1 channel
        
        for arch_layer in c2d_layers:
            self.conv2d_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=arch_layer.filters,
                    kernel_size=arch_layer.window_size,
                    stride=arch_layer.strides,
                    padding=0,
                )
            )
            in_channels = arch_layer.filters
        
        # Calculate the size after convolutions for the dense layer
        # This will need to be computed based on input size and conv operations
        self.fc_input_size = self._calculate_fc_input_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)
        
    def _calculate_fc_input_size(self, input_size: int) -> int:
        # Create a dummy input to calculate the size after convolutions
        x = torch.zeros(1, 1, input_size)  # batch_size=1, channels=1, length=input_size
        
        # Apply 1D convolutions
        for conv1d in self.conv1d_layers:
            x = F.relu(conv1d(x))
        
        # Reshape for 2D convolutions (add spatial dimension)
        # Based on the original reshape (61, 257, 1)
        batch_size, channels, length = x.shape
        x = x.view(batch_size, 1, 61, 257)  # reshape to 2D
        
        # Apply 2D convolutions
        for conv2d in self.conv2d_layers:
            x = F.relu(conv2d(x))
        
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Apply 1D convolutions with ReLU activation
        for conv1d in self.conv1d_layers:
            x = F.relu(conv1d(x))
        
        # Reshape for 2D convolutions
        batch_size, channels, length = x.shape
        x = x.view(batch_size, 1, 61, 257)  # reshape to 2D
        
        # Apply 2D convolutions with ReLU activation
        for conv2d in self.conv2d_layers:
            x = F.relu(conv2d(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # sigmoid activation for final output
        
        return x


def get_model(
    model_name: str, inputs: int, outputs: int, data_format: str = "channels_last"
) -> E2EModel:
    return E2EModel(
        n_outputs=outputs,
        c1d_layers=cE2E_1d_layers,
        c2d_layers=cE2E_2d_layers,
        input_size=inputs,
    )


if __name__ == "__main__":
    from models.app import train_model
    from models.runner import standard_run_parser

    # Get a standard parser, and the arguments out of it
    parser = standard_run_parser()
    args = parser.parse_args()
    setup = vars(args)

    # Actually train the model
    train_model(model_callback=get_model, **setup)
