import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import lightning as L


class SimpleCNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()

        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        out = F.log_softmax(out, dim=1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.test_accuracy.update(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer
