from models.SimpleNet import SimpleModel
from models.SimpleCNN import SimpleCNN
from datamodules.MNIST_datamodule import MNISTDataModule
from datamodules.FashionMNIST_datamodule import FashionMNISTDataModule
from train import train
from src.misc_functions import get_pretrained_guesses, apply_default_transform
from datetime import datetime
import os
from torchvision import models
import torch
from visualise import cam_on_image, run_gradcam_on_dir, run_scorecam_on_dir
from PIL import Image

"""
This main.py is just a placeholder to test the backbone!
The final version will be much cleaner and won't have an absurd amount of variables.
These variables are placeholders and will be set via the CLI or a config file once that functionality is implemented.
"""

model = models.resnet50(pretrained=True)
pretrained_model = True
datamodule = None
img = None
imgpath = "data/Custom_test_images/goose_mango.png"
# n_samples = 5
train_model = False
exp_name = 'test experiment/'
save_results = True
fname = exp_name + "goose mango resnet "
camtype = "gradcam"

if not exp_name:
    dt_string = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
    exp_name = "Experiment " + dt_string

if save_results:
    if not os.path.exists("results/" + exp_name):
        os.makedirs("results/" + exp_name)

if train_model:
    train(model, datamodule)

if imgpath:
    img = Image.open(imgpath).convert('RGB')
elif not datamodule:
    print("Warning: No test image or datamodule found")

model.eval()

targets = [273, 225, 207, 208]


# sample_labels = generate_sample_images(5, datamodule, out_foldername)

cam_on_image(camtype, img, targets, fname, model)
