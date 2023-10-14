import glob
from src.misc_functions import save_class_activation_images, preprocess_image, apply_default_transform, \
    get_pretrained_guesses
from PIL import Image
from src.gradcam import GradCam
from src.scorecam import ScoreCam
import torch


def prep_cam(camtype, model, target_layer, prep_img, target_class):
    if camtype.lower() == "gradcam":
        grad_cam = GradCam(model, target_layer)
        cam = grad_cam.generate_cam(prep_img, target_class)
    elif camtype.lower() == "scorecam":
        score_cam = ScoreCam(model, target_layer)
        cam = score_cam.generate_cam(prep_img, target_class)
    # elif camtype.lower() == "layercam":
    #    layer_cam = LayerCam(model, target_layer)
    #    cam = layer_cam.generate_cam(prep_img, target_class)
    else:
        print("Type not recognised")
        exit()
    return cam


def run_cam_on_directory(cam_type, sample_path, sample_labels, grad_model, target_layer, output_dir):
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
        file_name_to_export = img_name + ' ' + cam_type

        if type(target_layer) == int:
            cam = prep_cam(cam_type, grad_model, target_layer, prep_img, target_class)
            # Save mask
            save_class_activation_images(image, cam, file_name_to_export, output_dir)
        elif type(target_layer) == list:
            for i in range(len(target_layer)):
                cam = prep_cam(cam_type, grad_model, target_layer[i], prep_img, target_class)
                file_name = file_name_to_export + " layer " + str(target_layer[i])
                save_class_activation_images(image, cam, file_name, output_dir)
        else:
            print("Target layer must be int or list of ints")
            print("Type not recognised")
            exit()
    print('Grad cam completed')


def cam_on_image(camtype, image, targets, outname, model, target_layer):
    """
    :param camtype: (str) "gradcam" or "scorecam"
    :param image: (PIL image) image on which cam is to be performed
    :param targets: (list, int) (optional) target class(es) for activation map. If empty, will default to target with highest confidence
    :param outname: output filename
    :param model: (LightningModule) the model to test
    :param target_layer: (int or list of ints): target layer to hook the visualiser onto
    :return: Nothing
    """
    prep_img = preprocess_image(image)
    file_name_to_export = outname
    img_t = apply_default_transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)
    get_pretrained_guesses(out)

    # Results might be better if the final convolutional sublayer in layer4 can be isolated
    if type(target_layer) == int:
        if targets:
            for target in targets:
                cam = prep_cam(camtype, model, target_layer, prep_img, target)
                file_name = file_name_to_export + " target: " + str(target) + ' layer: ' + str(target_layer)
                save_class_activation_images(image, cam, file_name)
            else:
                cam = prep_cam(camtype, model, target_layer, prep_img, target_class=None)
                file_name = file_name_to_export + ' layer: ' + str(target_layer)
                save_class_activation_images(image, cam, file_name)
    elif type(target_layer) == list:
        for i in range(len(target_layer)):
            if targets:
                for target in targets:
                    cam = prep_cam(camtype, model, target_layer, prep_img, target)
                    file_name = file_name_to_export + " target: " + str(target) + ' layer: ' + str(target_layer[i])
                    save_class_activation_images(image, cam, file_name)
                else:
                    cam = prep_cam(camtype, model, target_layer, prep_img, target_class=None)
                    file_name = file_name_to_export + ' layer: ' + str(target_layer)
                    save_class_activation_images(image, cam, file_name)
