import lightning as L
from lightning.pytorch.loggers import CSVLogger
from models.SimpleNet import SimpleModel
from models.SimpleCNN import SimpleCNN

from datamodules.MNIST_datamodule import MNISTDataModule
from datamodules.FashionMNIST_datamodule import FashionMNISTDataModule

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

model = simple_model()
datamodule = FashionMNISTDataModule()

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    #max_epochs=10,
    #callbacks=[TQDMProgressBar(refresh_rate=20)],
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    logger=CSVLogger(save_dir="logs/"),
    #limit_train_batches=10,
    #limit_test_batches=10
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)