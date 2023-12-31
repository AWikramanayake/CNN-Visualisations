import glob
from src.misc_functions import save_class_activation_images, preprocess_image, apply_default_transform, get_pretrained_guesses
from PIL import Image
from src.gradcam import GradCam
from src.scorecam import ScoreCam
import torch


def run_gradcam_on_dir(sample_path, sample_labels, grad_model, output_dir):
    """
    :param sample_path: (str) directory with images on which gradcam is to be run
    :param sample_labels: (list) (optional) list of labels/target classes. If None, activation map will be performed with label of highest confidence
    :param grad_model: (lightningModule) the model to test
    :param output_dir: (str) (optional) directory to output cam heatmaps to. If empty, images will be written into the source directory
    :return: Nothing
    """
    if not output_dir:
        output_dir = sample_path
    img_list = glob.glob(sample_path + "/*.png")

    for i in img_list:
        img_name = (i.split("/")[-1]).split(".")[0]
        print(img_name)
        image = Image.open(sample_path + '/' + img_name + '.png').convert('RGB')
        if sample_labels:
            target_class = sample_labels[i]
        else:
            target_class = None
        prep_img = preprocess_image(image)
        file_name_to_export = img_name + ' gradcam'

        grad_cam = GradCam(grad_model, target_layer=4)
        # Generate cam mask
        cam = grad_cam.generate_cam(prep_img, target_class)
        # Save mask
        save_class_activation_images(prep_img, cam, file_name_to_export, output_dir)
    print('Grad cam completed')


def run_scorecam_on_dir(sample_path, grad_model, output_dir=None):
    """
    :param sample_path: (str) directory with images on which scorecam is to be run
    :param sample_labels: (list, int) (optional) list of labels/target classes. If None, activation map will be performed with label of highest confidence
    :param grad_model: (lightningModule) the model to test
    :param output_dir: (str) (optional) directory to output cam heatmaps to. If empty, images will be written into the source directory
    :return: Nothing
    """
    if not output_dir:
        output_dir = sample_path
    img_list = glob.glob(sample_path + "/*.png")

    for i in img_list:
        img_name = (i.split("/")[-1]).split(".")[0]
        print(img_name)
        image = Image.open(sample_path + '/' + img_name + '.png').convert('RGB')
        target_class = None
        prep_img = preprocess_image(image)
        file_name_to_export = img_name + ' scorecam'

        score_cam = ScoreCam(grad_model, target_layer=4)
        # Generate cam mask
        cam = score_cam.generate_cam(prep_img, target_class)
        # Save mask
        save_class_activation_images(prep_img, cam, file_name_to_export, output_dir)
    print('Score cam completed')


def cam_on_image(camtype, image, targets, outname, model):
    """
    :param camtype: (str) "gradcam" or "scorecam"
    :param image: (PIL image) image on which cam is to be performed
    :param targets: (list, int) (optional) target class(es) for activation map. If empty, will default to target with highest confidence
    :param outname: output filename
    :param model: (LightningModule) the model to test
    :return: Nothing
    """
    prep_img = preprocess_image(image)
    file_name_to_export = outname
    img_t = apply_default_transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)
    get_pretrained_guesses(out)

    # Results might be better if the final convolutional sublayer in layer4 can be isolated
    if camtype.lower() == "gradcam":
        cam_init = GradCam(model, target_layer=4)
    elif camtype.lower() == "scorecam":
        cam_init = ScoreCam(model, target_layer=4)
    else:
        print("Cam type not recognised")
        exit()

    if targets:
        for target in targets:
            cam = cam_init.generate_cam(prep_img, target)
            file_name = file_name_to_export + " target: " + str(target)
            save_class_activation_images(image, cam, file_name)
    else:
        cam = cam_init.generate_cam(prep_img, None)
        save_class_activation_images(image, cam, file_name_to_export)
    print('Cam completed')
