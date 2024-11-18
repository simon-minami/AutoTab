'''
run training according to tabcnn paper
'''
import torch
from torch.utils.data import DataLoader
from GuitarSet import GuitarSet
from TabCNN import TabCNN
from Trainer import Trainer
from torch.utils.data import random_split

if __name__ == '__main__':
    generator = torch.Generator().manual_seed(42)

    batch_size = 128
    dataset = GuitarSet()
    train_dataset, val_dataset, test_dataset = random_split(dataset, (0.8, 0.1, 0.1), generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = TabCNN()
    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=1.0)

    trainer = Trainer(
        epochs=20,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer
    )

    trainer.fit()
