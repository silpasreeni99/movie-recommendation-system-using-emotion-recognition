# Project Name : Movie recommendation system using emotion recognition



# Description:

	Choosing the right movie to watch is not an easy job. We maybe in the state of mind to watch something which makes us feel good, or may want to watch some melodrama due to our sober state of mind, or may want to watch some movie which helps us to connect with music and relieve ourselves or maybe just watch something such that we relate to the characters on an emotional level. 
	We can’t settle upon a decision by just assuming what kind of emotion we are experiencing and choose something to watch based on that. 
	This is the main reason for the creation of a movie recommendation system which suggests apt movies to the user based on the user’s current emotion. This will help in making the user’s movie experience, a better one.
	The major factor for movie recommendation will be the user’s current emotion, which will be recognized through an automated process.

## What does Emotion Recognition mean?

	Emotion Recognition is the process of recognizing the emotions on human face by using advanced image processing techniques.


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


# Web scraping process
	The process of extracting useful information and data from a webpage by accessing the HTML of it is called as web scraping.
	So for this project, we are using the web scraping process to segregate information from the website by using python and BeautifulSoup. 
	We use web scarping process for the movie recommendation part






Hope you guys are able to use the code I have provided and explore a wonderful project!!
