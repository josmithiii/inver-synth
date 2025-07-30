import h5py
import numpy as np
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset


class SoundDataGenerator(Dataset):
    "Generates data for PyTorch"

    def __init__(
        self,
        data_file=None,
        batch_size=32,
        n_samps=16384,
        shuffle=True,
        last: float = 0.0,
        first: float = 0.0,
        channels_last=False,
        for_autoencoder=False,
    ):
        "Initialization"
        self.dim = (1, n_samps)
        self.batch_size = batch_size
        self.shuffle_enabled = shuffle  # Renamed from self.shuffle to avoid naming conflict
        self.data_file = data_file
        self.n_channels = 1
        self.for_autoencoder = for_autoencoder
        # For the E2E model, need to return channels last?
        if channels_last:
            self.expand_axis = 2
        else:
            self.expand_axis = 1

        database = h5py.File(data_file, "r")

        self.database = database

        self.n_samps = self.read_file(0).shape[0]
        print("N Samps in audio data: {}".format(self.n_samps))

        # set up list of IDs from data files
        n_points = len(database["files"])
        self.list_IDs = range(len(database["files"]))

        print(f"Number of examples in dataset: {len(self.list_IDs)}")
        slice: int = 0
        if last > 0.0:
            slice = int(n_points * (1 - last))
            self.list_IDs = self.list_IDs[slice:]
            print(f"Taking Last N points: {len(self.list_IDs)}")
        elif first > 0.0:
            slice = int(n_points * first)
            self.list_IDs = self.list_IDs[:slice]
            print(f"Taking First N points: {len(self.list_IDs)}")

        # set up label size from data files
        self.label_size = len(database["labels"][0])
        self.on_epoch_end()

    def get_audio_length(self):
        return self.n_samps

    def get_label_size(self):
        return self.label_size

    def __len__(self):
        "Denotes the number of samples in the dataset"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generate one sample of data"
        # Get the actual index from our shuffled list
        actual_index = self.indexes[index]
        
        # Read labels
        y = self.database["labels"][actual_index]
        
        # Load soundfile data
        data = self.read_file(actual_index)
        if data.shape[0] > self.n_samps:
            print(
                "Warning - too many samples: {} > {}".format(
                    data.shape[0], self.n_samps
                )
            )
        x = data[: self.n_samps]
        
        # Convert to tensors and add channel dimension
        x = torch.FloatTensor(x).unsqueeze(0)  # Add channel dimension (1, n_samps)
        y = torch.FloatTensor(y)
        
        if self.for_autoencoder:
            return y, y
        return x, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle_enabled is True:  # Use shuffle_enabled instead of shuffle
            np.random.shuffle(self.indexes)

    # Think this makes things worse - fills up memory
    # @lru_cache(maxsize=150000)
    def read_file(self, index):
        filename = self.database["files"][index]
        fs, data = wavfile.read(filename)
        return data

    def shuffle(self):
        "Shuffle the dataset - call this at the start of each epoch"
        if self.shuffle_enabled:  # Use shuffle_enabled instead of shuffle
            np.random.shuffle(self.indexes)
