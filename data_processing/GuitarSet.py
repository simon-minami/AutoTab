'''
guitarset is a pytorch dataset object
used to load and process data into windows for training
used AFTER audio preprocessed by AudioPreprocessor
'''
from torch.utils.data import Dataset
import torch
import os
import numpy as np
class GuitarSet(Dataset):
    def __init__(self, data_path='guitarset/spec_repr', window_size=9, filenames=None):
        # Initialize the dataset object with the data path and window size
        self.data_path = data_path
        self.window_size = window_size
        self.half_window_size = self.window_size // 2  # Half of the context window size for padding

        # Load and sort file paths from the specified directory
        # default is to load all audio files from data_path
        if filenames is None:
            self.filenames = sorted(os.listdir(self.data_path))
        else:
            self.filenames = [filename + '.npz' for filename in filenames]

        self.file_paths = [os.path.join(data_path, f) for f in self.filenames]




        # Initialize lists to store processed audio and label data
        audio_data = list()
        label_data = list()

        # Process each audio file individually
        for path in self.file_paths:
            data = np.load(path)  # Load the data from the .npz file

            audio = data['audio']  # Extract the audio representation
            num_frames = audio.shape[0]  # Number of frames in the audio
            # Apply padding to the audio to accommodate context windows at the start and end
            # padded_audio = np.pad(audio, [(self.half_window_size, self.half_window_size), (0, 0)], mode='constant')

            padded_audio = np.pad(audio, [(0, 0), (self.half_window_size, self.half_window_size)], mode='constant')
            # shape should be (192, padded_frames)
            # add a channel dimension to match the PyTorch format (1, 192, num_frames)
            padded_audio = torch.tensor(padded_audio, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 192, padded_frames)

            # Extract sliding windows across the frame dimension using unfold
            X = padded_audio.unfold(dimension=2, size=self.window_size, step=1).permute(2, 0, 1, 3)
            # Shape of X: (num_frames, 1, 192, window_size) - ready to be used by PyTorch saved_models
            audio_data.append(X)  # Add the processed audio windows to the list

            # Convert the labels to a tensor and add to the label list
            y = torch.tensor(data['labels'], dtype=torch.long)
            label_data.append(y)

        # Concatenate all audio data and label data along the frame dimension
        # audio_data should be a list of tensors with shape (frames, 1, 192, window_size)
        self.audio = torch.cat(audio_data, dim=0)  # Resulting shape: (total_frames, 1, 192, window_size)

        # label_data should be a list of tensors with shape (frames, 6)
        self.labels = torch.cat(label_data, dim=0)  # Resulting shape: (total_frames, 6)

        # Display final shapes for verification
        print(f'all audio shape: {self.audio.shape}')
        print(f'all labels shape: {self.labels.shape}')
    def __len__(self):
        return self.audio.shape[0]
    def __getitem__(self, idx):
        X = self.audio[idx]
        y = self.labels[idx]
        # print(f'returning X, y shape: {X.shape, y.shape}')
        return X, y

## test
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Initialize the dataset
    guitarset = GuitarSet(data_path='../guitarset/spec_repr', window_size=9)

    # Wrap it in a DataLoader
    dataloader = DataLoader(guitarset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over batches
    for X_batch, y_batch in dataloader:
        print(f'Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}')
        # Proceed with training or evaluation


