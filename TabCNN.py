# first, model design
import torch.nn as nn

'''
some notes: what operation does conv 2d actually do? it takes sum of element wise multiplication
'''
import torch
import torch.nn as nn

class TabCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (1, 192, 9)

        # First convolutional block: (1, 192, 9) -> (32, 190, 7)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 3x3 kernel, no padding, stride=1 (default)
            nn.ReLU()  # Activation function
        )

        # Second convolutional block: (32, 190, 7) -> (64, 188, 5)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU()
        )

        # Third convolutional block: (64, 188, 5) -> (64, 186, 3)
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        # Max-pooling layer: (64, 186, 3) -> (64, 93, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)  # 2x2 kernel, stride=2 (default)

        # Dropout after pooling
        self.dropout_1 = nn.Dropout(0.25)

        # Fully connected layers
        # Flatten the output of the max-pool: (64, 93, 1) -> (64 * 93)
        self.flatten = nn.Flatten()

        # First fully connected layer: (64 * 93) -> (128)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 93, out_features=128),
            nn.ReLU()
        )

        # Dropout after the first fully connected layer
        self.dropout_2 = nn.Dropout(0.5)

        # Second fully connected layer: (128) -> (126)
        self.fc2 = nn.Linear(in_features=128, out_features=126)

        # Softmax activation for the final output: reshaped to (6, 21)
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax row-wise (over fret classes)

    def forward(self, x):
        # Forward pass through convolutional blocks
        x = self.conv_block_1(x)
        # print(f'after conv 1: {x.shape}')
        x = self.conv_block_2(x)
        # print(f'after conv 2: {x.shape}')
        x = self.conv_block_3(x)
        # print(f'after conv 3: {x.shape}')

        # Max-pooling and dropout
        x = self.max_pool(x)
        # print(f'after max pool: {x.shape}')
        x = self.dropout_1(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        # print(f'after flatten: {x.shape}')
        x = self.fc1(x)
        # print(f'after fc1: {x.shape}')
        x = self.dropout_2(x)
        x = self.fc2(x)
        # print(f'after fc2: {x.shape}')

        # Reshape the output to (6, 21) and apply softmax row-wise
        x = x.view(-1, 6, 21)  # Reshape to (batch_size, 6 strings, 21 fret classes)
        # print(f'after reshape: {x.shape}')
        # x = self.softmax(x)  # loss fn expects raw logits

        return x

if __name__ == '__main__':
    # testing
    from torch.utils.data import DataLoader
    from GuitarSet import GuitarSet

    lebron = GuitarSet()
    dataloader = DataLoader(lebron, batch_size=32)
    lonzo = TabCNN()
    for X_batch, y_batch in dataloader:
        print(f'X_batch shape: {X_batch.shape} | y_batch shape: {y_batch.shape}')
        with torch.inference_mode():
            result = lonzo(X_batch)
        print(f'result: {result.shape}')
        print(torch.sum(result[0], dim=1))
        print(f'example y: {y_batch[30]}')
        break


