# CNN-Visualisations

The goal of this project is to create a robust framework to streamline the process of using XAI visualisation techniques to qualitatively study the performance of CNNs.
</br>

## Visualising the Performance of a CNN: a Quick Introduction

Consider the task of image classification using a CNN. For example, using a VGG16 model pretrained on the ImageNet dataset to label this image:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/misc_assets/goose%20mango%20base.png" width="650"/>
</p>
<p align="center">
Fig 1: Goose (left) and Mango (right)</br>
Clarification: "Goose" and "Mango" are their names, NOT the target classes for classification!
</p>
</br>

Here are VGG16's top 30 guesses for a suitable label:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/misc_assets/goose%20mango%20vgg16%20predictions.png" width="650" />
</p>
<p align="center">
  VGG16 model predictions for the image. </br>
  The format: label rank, label ID, label description, % confidence (relative).
</p>
</br>

As you can see, VGG16's top guess is correct: Mango is indeed a Labrador retreiver. But suppose we wanted to look deeper into *how* VGG16 makes these guesses. One way we could do that is to use explainable AI (XAI) methods such as [GradCam](https://arxiv.org/abs/1610.02391) to identify which parts of the image have the biggest impact on the classification score.

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/misc_assets/goose%20mango%20vgg%20target_%20208_Cam_On_Image_672p.png" />
</p>
<p align="center">
  Left: Heatmap of regions contributing to classification label 208, "Labrador retreiver".</br>
  Right: Heatmap of regions contributing to classification label 285, "Egyptian cat".
</p>
</br>


Here, [ScoreCam](https://arxiv.org/abs/1910.01279) has been used to study what VGG16 is looking at. The hierachy of the heatmap is blue -> red. For the label "Labrador retreiver", VGG16's attention is indeed on mango, in particular, her snout, which is indeed a very dog-ish feature. Makes perfect sense.</br>
For the label "Egyptian cat" (VGG16's 14th most confident guess), while some regions around mango are still active, VGG16's attention is correctly largely shifted towards Goose.</br>

But wait, VGG16's 12th most confident guess, and it's most confident non-dog guess, was "dishwasher"! What's going on there? There clearly isn't a dishawasher anywhere in the image, so let's examine where VGG16 thinks it can see one:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/misc_assets/goose%20mango%20vgg%20target_%20534_Cam_On_Image%20and%20zoom.png" />
</p>
<p align="center">
  Left: Heatmap of regions contributing to classification label 534, "Dishwasher".</br>
  Right: Zoom-in of the active region.
</p>
</br>


Huh. Interesting. Zooming in, the reflection of the light in the window does kind of look like a dish in a dishwasher rack, doesn't it? The way the CNN looks at details a human observer would've skimmed over, and the way it mistakes a reflection for an object is an interesting look into how the model sees things.
</br></br>

## Practical applications of XAI

That's all well and good, but does XAI *really* have a concrete use in studying models beyond creating maps that are *kind of interesting*? Can XAI really be used to improve model performance? Yes, it does!

Consider [this Taiwanese study](https://www.mdpi.com/1660-4601/18/3/961) that used CNNs to perform species recognition tasks on trees. XAI tools helped researchers identify one of the causes of the model's unexpectedly poor performace: too many of the pictures in their dataset contained buildings in the background, causing the model to learn to focus on the buildings rather than the trees in some cases.

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/misc_assets/CNN%20Bias%20example.png" width="650"/>
</p>
<p align="center">
  Figure from the linked study demonstrating how XAI visualisations were used to show that the model was incorrectly focusing on the buildings instead of the trees, and showing that the issue has been mitigated in the proposed model.</br>
  Note: the heatmap colours are reversed compared to the images above.
</p>
</br></br>


With this insight in hand, the researchers were able to work on fixing this bias in their data.
</br></br>

## Where This Project Comes In

The above case is a classic example of the type of bias that can slip through the cracks when training a CNN. The goal of this project is to create a robust framework that can streamline the process of apllying XAI visualisations to study CNNs so that issues like this can be easily spotted.

The intended first step is to automate the process of applying the visualisations in [this fantastic repository](https://github.com/utkuozbulak/pytorch-cnn-visualizations). The idea is to evaluate the model on a test set, automatically pick out samples (random or cases with incorrect guesses and/or low confidence guesses), and perform the selected XAI visualisations on them. The goal is to ensure compatibility with as few restrictions on the model/datamodule as possible, and to ensure compatibility with a wide range of models in the torchvision model zoo.

Beyond that, the next goals are to implement more XAI techniques and to allow the experiment parameters to be controlled from the CLI.
</br></br>

### Project Updates:

* alexnet, vgg16, and resnet (and possibly others with same/similar model structures) now work
* GradCam and ScoreCam are currently being implemented.
* The visualisations can be applied to random samples drawn from a suitably constructed datamodule.
</br></br>

### Updates in progress:

* Modifying the forward loop to ensure compatibility with more pretrained models (AlexNet, VGG, ResNet. Others to follow)
* Selectively applying visualisations to incorrect and low confidence guesses by the model.
