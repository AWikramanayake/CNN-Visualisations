"""
Originally created on Thu Oct 21 11:09:09 2017
by Utku Ozbulak - github.com/utkuozbulak

Modified for this repository by Akshath Wikramanayake
        - added generate sample to create images from the datamodule
        - modified preprocess_image to use mean and std from the datamodule for transforms if they exist
        - added generate samples to generate sample images from a datamodule
        - added get_pretrained_guesses to get the top n guesses/confidences from pretrained models
        - added apply_default_transform to facilitate getting pretrained guesses
"""
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('/results'):
        os.makedirs('/results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('/results', file_name + '.png')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name, out_dir=None):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
        out_dir: (str): OPTIONAL path to output directory to save cam images
    """
    if out_dir:
        output_dir = out_dir
    else:
        output_dir = 'results/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    # path_to_file = os.path.join(output_dir, file_name + '_Cam_Heatmap.png')
    # save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join(output_dir, file_name + '_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    # path_to_file = os.path.join(output_dir, file_name + '_Cam_Grayscale.png')
    # save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    org_im = org_im.resize((224, 224))
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def apply_heatmap(R, sx, sy):
    """
        Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial

        This is (so far) only used for LRP
    """
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heatmap
    # plt.show()


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)
    print("image saved to: " + path)


def preprocess_image(pil_im, resize_im=True, source_dm=None):
    """
        Processes image for CNNs
        Modified by Akshath Wikramanayake:
    Args:
        pil_im (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
        source_dm (datamodule): (If applicable) Datamodule from which images are taken, to get mean/std
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """

    if hasattr(source_dm, 'mean') and hasattr(source_dm, 'std'):
        mean = source_dm.mean
        std = source_dm.std
    else:
        # Mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def generate_sample_images(num_samples, source_dm, outpath):
    """
        Generates sample images from datamodule
    Args:
        num_samples (int): number of samples to generate
        source_dm (DataModule): Datamodule from which to generate samples
        outpath (string): output folder path for generated samples
    returns:

    """
    source_dm.setup()
    labels = []

    for i in range(num_samples):
        idx = np.random.randint(low=0, high=len(source_dm.test))
        example = source_dm.test[idx]
        filename = outpath + '/Sample ' + str(i) + ".png"
        torchvision.utils.save_image(example[0], fp=filename)
        labels.append(example[1])

    return labels


def get_pretrained_guesses(model_out, num_guesses=10):
    """
        Obtains the top label guesses of a model trained on the ImageNet dataset
    Args:
        model_out (tensor): model eval output
        num_guesses (int): number of top guesses to output
    returns:

    """
    with open('imagenet1000_clsidx_to_labels.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    percentage = torch.nn.functional.softmax(model_out, dim=1)[0] * 100

    _, indices = torch.sort(model_out, descending=True)
    pos = 1

    for idx in indices[0][:num_guesses]:
        print(str(pos) + ': ' + classes[idx] + str(percentage[idx].item()))
        pos += 1


def apply_default_transform(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    return transform(image)