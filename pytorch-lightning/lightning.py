import torch 
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F 

import pytorch_lightning as pl
from pytorch_lightning import Trainer


# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.validation_losses = []

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activatin and no softmax at the end
        return out 
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),
                                                download=True)
        
        # Dataloader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        return train_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.validation_losses.append(loss)
        return {'val_loss': loss}
    
    def val_dataloader(self):
        # MNIST dataset
        val_dataset = torchvision.datasets.MNIST(root='./data',
                                                train=False,
                                                transform=transforms.ToTensor())
                
        # Dataloader
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return val_loader
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_losses).mean()
        tensorboard_logs = {'tavg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == "__main__":
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)