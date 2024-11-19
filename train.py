'''
run training according to tabcnn paper
'''
import torch
from torch.utils.data import DataLoader
from GuitarSet import GuitarSet
from TabCNN import TabCNN
from Trainer import Trainer
from torch.utils.data import random_split
import argparse
# TODO see what output is on custom audio


def parser():
    parser = argparse.ArgumentParser(
        prog='tabcnn',
        description='training tabcnn',
        epilog='my goat lebron'
    )
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_audio_files', type=int, default=None, help='number of audio files to use')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser()
    generator = torch.Generator().manual_seed(42)

    batch_size = args.batch_size
    dataset = GuitarSet(num_files=args.num_audio_files)
    train_dataset, val_dataset, test_dataset = random_split(dataset, (0.8, 0.1, 0.1), generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = TabCNN()
    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=1.0)

    trainer = Trainer(
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer
    )

    trainer.fit()
    trainer.visualize()
    mp_precision = trainer.eval()
    print(mp_precision)
