# Facial Emotion Recognition
===============

This repository was used for my bachelor thesis to prove that it is possible to recognize emotions with a deep learning program. The aim of the project was to train a convolutional neutral network and test it on real cases. To train this network I used the FER-2013 dataset. This program is able to recognize the seven basic emotions angry, disgust, fear, happy, sad surprise and neutral.


Repository Contents
============
* **/dataset** - This folder is empty. Add your dataset
* **/img** - This is where all images are stored.
* **/models** - Here your trained models will be saved
* **/notebooks** - For data visualization I used this folder.
* **/src** - In this folder is the source code
* **/LICENSE** - The license file.
* **/README.md** - The file you're reading now!


Requirements
============
To get this project running you need to install following libraries
* python
* pandas
* numpy
* OpenCV
* scikit-learn
* TensorFlow / Theano
* Keras
* (Jupyter notebooks)
* FER-2013 dataset


Dataset
=======
The FER-2013 dataset consists of 35'887 prepared 48x48 pixel grayscale images of faces. They are labelled with one of the seven emotions.

* 4953 angry faces
* 547 disgust faces
* 5121 fear faces
* 8989 happy faces
* 6077 sad faces
* 4002 surprised faces
* 6198 neutral faces


Model
=====
My model consists of an input layer followed by Conv2D, Relu, Dropout and then Conv2D, MaxPool and then again Conv2D, Relu, Dropout. At the end I used two fully-connected layers and an output-layer.

![alt text][pic1]

[pic1]: https://github.com/Risnar/Facial-Emotion-Recognition/blob/master/img/CNN-Model.png "My CNN-Model"


Model Validation
================
When I trained this network I reached an accuracy of 72.89% on the train-set and an accuracy of 68.57% on the test-set.
![alt text][pic2]

[pic2]: https://github.com/Risnar/Facial-Emotion-Recognition/blob/master/img/graphs.png "Loss and Accuracy Diagram"


But let's have a closer look for each category. The prediction of my model varies from emotion to emotion, because the dataset didn't have the same amount of image for each emotion and some of the emotions are more difficult to differentiate.

![alt text][pic3]

[pic2]: https://github.com/Risnar/Facial-Emotion-Recognition/blob/master/img/confusionMatrix_val.png "Confusion Matrix"

As can be seen, the predictions for happy with 87.60% and Surprise with 80.75% have the highest accuracy. The lowest value was reached with 48.73% for fear.


Build Instructions
==================

To train this CNN you have to meet the requirements. Download the FER-2013 dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data "FER-2013 challenge"). Add it to the folder "dataset" and start your pyton environment. Then you can run the emotionRecogntion.py.

After training your model you can validate it on the liveDemo with your camera. Adapt the name and the paths and then run liveDemo.py and play around.
