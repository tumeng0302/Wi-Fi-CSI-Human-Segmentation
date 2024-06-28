import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from data.dataset import PersonInWifiDataset
import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Wi-Fi CSI Human Segmentation')
    parser.add_argument('--data_root', type=str, default='/root/CSI_Dataset', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use for training')
    args = parser.parse_args()
    return args

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        amp, pha, mask, jhm, paf, img = batch
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Parse arguments
args = argparser()
# Data
# Create datasets
train_dataset = PersonInWifiDataset(data_root=args.data_root, split='train')
val_dataset = PersonInWifiDataset(data_root=args.data_root, split='val')
test_dataset = PersonInWifiDataset(data_root=args.data_root, split='test')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Training
model = LitModel()
# trainer = pl.Trainer(max_epochs=10)  # Set max_epochs and gpus as per your requirements
trainer = pl.Trainer(fast_dev_run=True, accelerator='gpu', devices=[1])
trainer.fit(model, train_loader, val_loader)

# Testing
trainer.test(model, test_loader)