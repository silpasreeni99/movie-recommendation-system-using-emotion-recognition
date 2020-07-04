from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        from bs4 import BeautifulSoup as SOUP 
        import re 
        import requests as HTTP 

        # Main Function for scraping 
        def main(emotion):
            em=emotion.lower()
            # IMDb Url for Drama genre of 
            # movie against emotion Sad 
            if(em == "sad"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
                

            # IMDb Url for Musical genre of 
            # movie against emotion Disgust 
            elif(em == "disgust"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Family genre of 
            # movie against emotion Anger 
            elif(em == "angry"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Thriller genre of 
            # movie against emotion Anticipation 
            elif(em == "neutral"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Sport genre of 
            # movie against emotion Fear 
            elif(em == "scared"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Thriller genre of 
            # movie against emotion Enjoyment 
            elif(em == "happy"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'

            
            

            # IMDb Url for Film_noir genre of 
            # movie against emotion Surprise 
            elif(em == "surprised"):
                print("EMOTION DETECTED:",em)
                urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'

            # HTTP request to get the data of 
            # the whole page 
            response = HTTP.get(urlhere) 
            data = response.text 

            # Parsing the data using 
            # BeautifulSoup 
            soup = SOUP(data, "lxml") 

            # Extract movie titles from the 
            # data using regex 
            title = soup.find_all("a", attrs = {"href" : re.compile(r'\/title\/tt+\d*\/')})
            title1 = soup.find_all("h3",{"class":"lister-item-header"})
            print("LIST OF APT MOVIES BASED ON USERS CURRENT EMOTION:")
            print(title1[0].text)
            rating = soup.find_all("div", {"class": "inline-block ratings-imdb-rating"})
            print("rating=",rating[0].text)
            print(title1[1].text)
            print("rating=",rating[1].text)
            print(title1[2].text)
            print("rating=",rating[2].text)
            print(title1[3].text)
            print("rating=",rating[3].text)
            print(title1[4].text)
            print("rating=",rating[4].text)
            print(title1[5].text)
            print("rating=",rating[5].text)
            print(title1[6].text)
            print("rating=",rating[6].text)
            print(title1[7].text)
            print("rating=",rating[7].text)
            print(title1[8].text)
            print("rating=",rating[8].text)
            print(title1[9].text)
            print("rating=",rating[9].text)
            return title1
                

        # Driver Function 
        if __name__ == '__main__': 

            #emotion = input("Enter the emotion: ") 
            a = main(label) 
            count = 0

            if(label == "Disgust" or label== "Anger"
                                                    or label=="Surprise"): 

                for i in a: 

                        # Splitting each line of the 
                        # IMDb data to scrape movies 
                    tmp = str(i).split('>;') 

                    if(len(tmp) == 3): 
                            print(tmp[1][:-3]) 

                    if(count > 13): 
                            break
                    count += 1
            else: 
                for i in a: 
                    tmp = str(i).split('>') 

                    if(len(tmp) == 3): 
                            print(tmp[1][:-3]) 

                    if(count > 11): 
                            break
                    count+=1

            cv2.destroyAllWindows()
            break

       

camera.release()
cv2.destroyAllWindows()
