# Project Name : Movie recommendation system using emotion recognition



# Description:

The main aim of my project is to suggest movies to the user based on the emotion recognized.


## What does Emotion Recognition mean?

Emotion recognition is a technique used in software that allows a program to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he/she is feeling and not just that but also showing the probabilities of mixed emotions a face could has.

# Installations:
-keras

-imutils

-cv2

-numpy

-tensorflow


# Usage:

The program will creat a window to display the scene capture by webcamera and a window representing the probabilities of detected emotions.

> Demo

python real_time_video.py

You can just use this with the provided pretrained model i have included in the path written in the code file, i have choosen this specificaly since it scores the best accuracy, feel free to choose any but in this case you have to run the later file train_emotion_classifier
## If you just want to run this demo instead of training the model from scaratch, the following content can be skipped
> Train

python train_emotion_classifier.py



# Dataset:

I have used [this](https://www.kaggle.com/c/3364/download-all) dataset

Download it and put the csv in fer2013/fer2013/

-fer2013 emotion classification test accuracy: 66%


# Credits
The emotion recognition part is inspired from [this](https://github.com/omar178/Emotion-recognition) great work.




Hope you guys are able to use the code I have provided and explore a wonderful project!!
