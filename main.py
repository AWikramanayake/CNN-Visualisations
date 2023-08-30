from models.SimpleNet import SimpleModel
from models.SimpleCNN import SimpleCNN
from datamodules.ImageNet_datamodule import ImageNetDataModule
from datamodules.FashionMNIST_datamodule import FashionMNISTDataModule
from train import train
from vis_backbone.misc_functions import generate_sample_images, save_class_activation_images, preprocess_image
from datetime import datetime
import os
from visualise import run_gradcam

dt_string = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
out_foldername = "results/Experiment " + dt_string
os.makedirs(out_foldername)

model = SimpleCNN()
datamodule = ImageNetDataModule()
n_samples = 5
train_model = False

if train_model:
    train(model, datamodule)

sample_labels = generate_sample_images(5, datamodule, out_foldername)

run_gradcam(out_foldername, sample_labels, model, datamodule)




