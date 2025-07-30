import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from models.common.architectures import layers_map

"""
The STFT spectrogram of the input signal is fed
into a 2D CNN that predicts the synthesizer parameter
configuration. This configuration is then used to produce
a sound that is similar to the input sound.
"""


"""Model Architecture"""
# @ paper:
# 1 2D Strided Convolution Layer C(38,13,26,13,26)
# where C(F,K1,K2,S1,S2) stands for a ReLU activated
# 2D strided convolutional layer with F filters in size of (K1,K2)
# and strides (S1,S2).


class SpectrogramCNN(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        arch_layers: list,
        input_size: int,
        n_fft: int = 512,  # Orig:128
        hop_length: int = 256,  # Orig:64
        sample_rate: int = 16000,
    ):
        super(SpectrogramCNN, self).__init__()

        # STFT transform
        self.spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,  # Power spectrogram
        )

        # Convert to dB scale
        self.amplitude_to_db = transforms.AmplitudeToDB()

        # 2D Convolutional layers
        self.conv2d_layers = nn.ModuleList()
        in_channels = 1  # spectrogram has 1 channel

        for arch_layer in arch_layers:
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
        self.fc_input_size = self._calculate_fc_input_size(
            input_size, n_fft, hop_length
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _calculate_fc_input_size(
        self, input_size: int, n_fft: int, hop_length: int
    ) -> int:
        # Create a dummy input to calculate the size after transformations
        x = torch.zeros(1, 1, input_size)  # batch_size=1, channels=1, length=input_size

        # Apply spectrogram transform
        x = self.spectrogram(x.squeeze(1))  # Remove channel dim for spectrogram
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)  # Add channel dim back

        # Permute dimensions to match paper format (time, freq)
        x = x.permute(0, 1, 3, 2)  # (batch, channel, time, freq)

        # Apply 2D convolutions
        for conv2d in self.conv2d_layers:
            x = F.relu(conv2d(x))

        return x.view(1, -1).size(1)

    def forward(self, x):
        # x shape: (batch_size, 1, audio_length)

        # Apply STFT spectrogram
        x = self.spectrogram(x.squeeze(1))  # Remove channel dim for spectrogram
        x = self.amplitude_to_db(x)  # Convert to dB scale
        x = x.unsqueeze(1)  # Add channel dim back

        # Permute dimensions to match paper format (time, freq)
        x = x.permute(0, 1, 3, 2)  # (batch, channel, time, freq)

        # Apply 2D convolutions with ReLU activation
        for conv2d in self.conv2d_layers:
            x = F.relu(conv2d(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # sigmoid activation for final output

        return x


"""
Standard callback to get a model ready to train
"""


def get_model(
    model_name: str, inputs: int, outputs: int, data_format: str = "channels_last"
) -> SpectrogramCNN:
    arch_layers = layers_map.get("C1")
    if model_name in layers_map:
        arch_layers = layers_map.get(model_name)
    else:
        print(
            f"Warning: {model_name} is not compatible with the spectrogram model. C1 Architecture will be used instead."
        )
    return SpectrogramCNN(
        n_outputs=outputs,
        arch_layers=arch_layers,
        input_size=inputs,
    )


if __name__ == "__main__":

    from models.app import train_model
    from models.runner import standard_run_parser

    # Get a standard parser, and the arguments out of it
    parser = standard_run_parser()
    args = parser.parse_args()
    setup = vars(args)

    # distinguish model type for reshaping
    setup["model_type"] = "STFT"

    # Actually train the model
    train_model(model_callback=get_model, **setup)
