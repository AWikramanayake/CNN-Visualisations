import glob
from src.misc_functions_debug import save_class_activation_images, preprocess_image
from PIL import Image
from src.gradcam import GradCam
from src.scorecam import ScoreCam


def run_gradcam_on_dir(sample_path, sample_labels, grad_model, dm):
    for i in range(len(glob.glob(sample_path + '/*.png'))):
        image = Image.open(sample_path + '/Sample ' + str(i) + '.png').convert('RGB')
        target_class = sample_labels[i]
        prep_img = preprocess_image(image, source_dm=dm)
        file_name_to_export = 'gradcam_test ' + str(i)

        grad_cam = GradCam(grad_model, target_layer=4)
        # Generate cam mask
        cam = grad_cam.generate_cam(prep_img, target_class)
        # Save mask
        save_class_activation_images(image, cam, file_name_to_export, sample_path)
    print('Grad cam completed')


def run_gradcam(input_img, fname, target_classes, grad_model):
    image = input_img
    prep_img = preprocess_image(image)

    for i in range(len(target_classes)):
        target_class = target_classes[i]
        grad_cam = GradCam(grad_model, target_layer=10)
        cam = grad_cam.generate_cam(prep_img, target_class)

        file_name_to_export = fname + "target: " + str(target_classes[i])

        save_class_activation_images(image, cam, file_name_to_export)


def run_scorecam(input_img, fname, target_classes, grad_model):
    image = input_img
    prep_img = preprocess_image(image)

    for i in range(len(target_classes)):
        target_class = target_classes[i]
        score_cam = ScoreCam(grad_model, target_layer=28)
        cam = score_cam.generate_cam(prep_img, target_class)

        file_name_to_export = fname + "target: " + str(target_classes[i])

        save_class_activation_images(image, cam, file_name_to_export)
