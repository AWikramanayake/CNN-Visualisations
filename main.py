from models.SimpleNet import SimpleModel
from models.SimpleCNN import SimpleCNN
from datamodules.MNIST_datamodule import MNISTDataModule
from datamodules.FashionMNIST_datamodule import FashionMNISTDataModule
from train import train
from vis_backbone.misc_functions import generate_sample_images, save_class_activation_images, preprocess_image
from vis_backbone.gradcam import GradCam
from datetime import datetime
import os
import glob
from PIL import Image

dt_string = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
out_foldername = "results/Experiment " + dt_string
os.makedirs(out_foldername)

model = SimpleCNN()
datamodule = FashionMNISTDataModule()
n_samples = 5
train_model = False

if train_model:
    train(model, datamodule)

sample_labels = generate_sample_images(5, datamodule, out_foldername)


for i in range(len(glob.glob(out_foldername + '/*.png'))):
    image = Image.open(out_foldername + '/Sample ' + str(i) + '.png').convert('L')
    target_class = sample_labels[i]
    prep_img = preprocess_image(image)
    file_name_to_export = 'gradcam_test ' + str(i)

    grad_cam = GradCam(model, target_layer=4)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(image, cam, file_name_to_export, out_foldername)
    print('Grad cam completed')



