# Mask Detection

These are examples of using the **TrackEverything** package, you can find instructions on installation and other explanations [here](https://github.com/ami-a/TrackEverything). These Mask Detection examples are open-sourced framework built on top of the **TrackEverything** package and uses detection models, classification models, tracking algorithms and statistics-based decision making. The project allows you to detected people with or without masks, I used several models from different repositories or packages and combined them.

## Overview

You can find all the models and test videos [here](https://drive.google.com/drive/folders/19jsLpv8Ql_ebqYZy1vnC3Snp0dNQ8HX0?usp=sharing).

### Mask Example 1

#### The Detection Model

This example uses an Head Detection model from [AVAuco/ssd_head_keras github repository](https://github.com/AVAuco/ssd_head_keras) for detecting heads, I modified the files to be compatible with TF2.2. The model has been trained using the [Hollywood Heads dataset](https://www.robots.ox.ac.uk/~vgg/software/headmview/) as positive samples, and a subsample of the [EgoHands dataset](http://vision.soic.indiana.edu/projects/egohands/) as negative
samples. This model has been developed using [Pierluigi Ferarri's Keras implementation of SSD](https://github.com/pierluigiferrari/ssd_keras/) as primary source, and replicates the original [Matconvnet version of our model](https://github.com/AVAuco/ssd_people). In the `custom_get_detection_array` I use the model to give me all the heads detected in a frame with a score of at least `detection_threshold=0.4`. Later I filter out redundant overlapping detections using the default Non-maximum Suppression (NMS) method. <p align="center"><img src="images/repos/head_det.jpg" width=540 height=404></p>

### The Classification Model

After we have the heads from the detection model, I put them through a classification model to determine the probability of them being with a mask. I used the Face Mask classification model from [chandrikadeb7/Face-Mask-Detection github repository](https://github.com/chandrikadeb7/Face-Mask-Detection). It's based on the MobileNetV2 architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.). <p align="center"><img src="images/repos/mask_class_1.png" width=540 height=328></p>

## Results

I only tested it on one video I found online but the results are fair and setting could be optimized much more. The head detection is very rudimentary and has a lot of misses and partial matches.
<p align="center"><img src="images/screens/mask01.png" width=564 height=337></p>

### Mask Example 2
#### The Detection Model

This example uses a Face Detection model from OpenCV for detecting faces. OpenCV ships out-of-the-box with pre-trained Haar cascades that can be used for face detection and a deep learning-based face detector that has been part of OpenCV since OpenCV 3.3. In the `custom_get_detection_array` I use OpenCV to give me all the faces detected in a frame with a score of at least `detection_threshold=0.12`. Later I filter out redundant overlapping detections using the default Non-maximum Suppression (NMS) method.

### The Classification Model

I used the same [classification model](#the-classification-model) as in [example 1](#mask-example-1).

## Results

The results are fair and better from example 1, mainly since the face detector is better. The classification model is not very good and has a lot of misses, but optimizing the detector's parameters can make better results.
<p align="center"><img src="images/screens/mask02.png" width=564 height=318></p>

### Mask Example 3

#### The Detection Model

This example uses a Mask Detection and Classification model from [PureHing/face-mask-detection-tf2 github repository](https://github.com/PureHing/face-mask-detection-tf2) for detecting faces and classify them. This model is a lightweight face mask detection model based on ssd and the backbone is MobileNet and RFB. Since this model also classifies there is no need for an additional classification model. In the `detection_vars.py` I use the model to give me all the heads detected in a frame with a score of at least `DETECTION_THRESHOLD=0.4` and later I filter out redundant overlapping detections using the default Non-maximum Suppression (NMS) method. I also receive classification data score from the model and input them as a vector for the detector. <p align="center"><img src="images/repos/mask_class_3.png" width=540 height=325></p>

## Results

I tested it on the same video I found online and the results are very good and the best so far. Changing the setting could help for receiving even greater results.
<p align="center"><img src="images/screens/mask03.png" width=564 height=337></p>