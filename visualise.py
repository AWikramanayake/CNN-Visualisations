import glob
from vis_backbone.misc_functions import save_class_activation_images, preprocess_image
from PIL import Image
from vis_backbone.gradcam import GradCam


def run_gradcam(sample_path, sample_labels, grad_model, dm):
    for i in range(len(glob.glob(sample_path + '/*.png'))):
        image = Image.open(sample_path + '/Sample ' + str(i) + '.png').convert('L')
        target_class = sample_labels[i]
        prep_img = preprocess_image(image, source_dm=dm)
        file_name_to_export = 'gradcam_test ' + str(i)

        grad_cam = GradCam(grad_model, target_layer=4)
        # Generate cam mask
        cam = grad_cam.generate_cam(prep_img, target_class)
        # Save mask
        save_class_activation_images(image, cam, file_name_to_export, sample_path)
    print('Grad cam completed')

