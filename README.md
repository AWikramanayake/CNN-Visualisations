# CNN-Visualisations

The goal of this project is to create a robust framework to streamline the process of using XAI visualisation techniques to qualitatively study the performance of CNNs.  
Please note that this project is very much a work in progress, so stay tuned for updates!

The intended first step is to automate the process of applying the visualisations in [this fantastic repository](https://github.com/utkuozbulak/pytorch-cnn-visualizations) by Utku Ozbulak.  
So far GradCam is largely operational and more techniques are soon to follow. Right now the visualisations can be applied to random samples drawn from a datamodule, and in the future the option to apply them selectively to incorrect or low-confidence guesses will be implemented.

Beyond that, the goals are to implement more XAI techniques and to allow the experiment parameters to be controlled from the CLI.

### Project Updates:

* GradCam visualisation is largely operational.
* The visualisations can be applied to random samples drawn from a datamodule.

### Updates in progress:

* Implementing more visualisation techniques.
* Selectively applying visualisations to incorrect and low confidence guesses by the model.
