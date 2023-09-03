# CNN-Visualisations

The goal of this project is to create a robust framework to streamline the process of using XAI visualisation techniques to qualitatively study the performance of CNNs.
</br>

## Visualising the Performance of a CNN: a Quick Introduction

Consider the task of image classification using a CNN. For example, using a VGG16 model pretrained on the ImageNet dataset to label this image:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/goose%20mango%20base.png" width="650"/>
</p>
<p align="center">
Fig 1: Goose (left) and Mango (right)</br>
Clarification: "Goose" and "Mango" are their names, NOT the target classes for classification!
</p>
</br>

Here are VGG16's top 30 guesses for a suitable label:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/goose%20mango%20vgg16%20predictions.png" width="650" />
</p>
<p align="center">
  VGG16 model predictions for the image (the numbers are the relative percentage confidence levels for a given label)
</p>
</br>

As you can see, VGG16's top guess is correct: Mango is indeed a Labrador retreiver! But suppose we wanted to look deeper into *how* VGG16 makes these guesses. One way we could do that is to use explainable AI (XAI) methods such as [GradCam](https://arxiv.org/abs/1610.02391) to identify which parts of the image have the biggest impact on the classification score.

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/goose%20mango%20vgg%20target_%20208_Cam_On_Image_672p.png" />
</p>
<p align="center">
  Left: Heatmap of regions contributing to classification label 208, "Labrador retreiver".</br>
  Right: Heatmap of regions contributing to classification label 285, "Egyptian cat".
</p>
</br>


Here, [ScoreCam](https://arxiv.org/abs/1910.01279) has been used to study what VGG16 is looking at. For the label "Labrador retreiver", VGG16's attention is indeed on mango. For the label "Egyptian cat" (VGG16's 14th most confident guess), VGG16's attention is correctly largely shifted towards Goose.

But wait, VGG16's 12th most confident guess, and it's most confident non-dog guess, was "dishwasher"! What's going on there? There clearly isn't a dishawasher anywhere in the image, so let's examine where VGG16 thinks it can see one:

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/goose%20mango%20vgg%20target_%20534_Cam_On_Image%20and%20zoom.png" />
</p>
<p align="center">
  Left: Heatmap of regions contributing to classification label 534, "Dishwasher".</br>
  Right: Zoom-in of the active region.
</p>
</br>


Huh. Interesting. Zooming in, the reflection of the light in the window does kind of look like a dish in a dishwasher rack, doesn't it? The way the CNN looks at details a human observer would've skimmed over, and the way it mistakes a reflection is an interesting look into how the model sees things.
</br></br>

## Practical applications of XAI

That's all well and good, but does XAI *really* have a concrete use in studying models beyond creating maps that are *kind of interesting*? Can XAI really be used to improve model performance? Yes, it does!

Consider [this Taiwanese study](https://www.mdpi.com/1660-4601/18/3/961) using CNNs to perform species recognition tasks on trees. XAI tools helped researchers identify the cause of poor performace: too many of the pictures in their dataset contained buildings in the background, causing the model to focus on the buildings in some cases instead!

<p align="center">
<img src="https://raw.githubusercontent.com/AWikramanayake/CNN-Visualisations/master/CNN%20Bias%20example.png" width="650"/>
</p>
<p align="center">
  Figure from the linked study demonstrating how XAI visualisations were used to show that the model was incorrectly focusing on the buildings instead of the trees, and showing that the issue has been mitigated in the proposed model.</br>
  Note: the heatmap colours are reversed compared to the images above.
</p>
</br></br>


With this insight in hand, the researchers were able to work on fixing this bias in their data.
</br></br>

## Where This Project Comes In

The above is a classic example of the type of bias that can slip through the cracks when using a CNN. The goal of this project is to create a robust framework that can streamline the process of apllying XAI visualisations to study the performance of a model.

The intended first step is to automate the process of applying the visualisations in [this fantastic repository](https://github.com/utkuozbulak/pytorch-cnn-visualizations), so that samples from the test set that are incorrectly labeled and/or labeled with low confidence can be automatically picked out and the corresponding guesses can be visualised.

Beyond that, the next goals are to implement more XAI techniques and to allow the experiment parameters to be controlled from the CLI.
</br></br>

### Project Updates:

* GradCam and ScoreCam are currently being implemented.
* The visualisations can be applied to random samples drawn from a suitably constructed datamodule.
</br></br>

### Updates in progress:

* Modifying the forward loop to ensure compatibility with more pretrained models (AlexNet, VGG, ResNet. Others to follow)
* Selectively applying visualisations to incorrect and low confidence guesses by the model.
