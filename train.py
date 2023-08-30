import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def train(model, datamodule):
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=10,
        #callbacks=[TQDMProgressBar(refresh_rate=20)],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        logger=CSVLogger(save_dir="logs/"),
        #limit_train_batches=10,
        #limit_test_batches=10
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)