from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import messagebox
root124= tk.Tk()
root124.withdraw()
msgbox=tk.messagebox.showinfo('HELLO SCREEN', "WELCOME TO MOVIE RECOMMENDATION SYSTEM BASED ON EMOTION RECOGNITION!!")

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
    else:
        import tkinter as tk
        from tkinter import messagebox
        root123= tk.Tk()
        root123.withdraw()
        msgbox=tk.messagebox.showinfo('ERROR MESSAGE', "FACE NOT DETECTED,PLEASE TRY AGAIN!!")
        break

 
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
            import tkinter as tk
            from tkinter import messagebox
            root= tk.Tk()
            root.withdraw()
            
            msgbox=tk.messagebox.showinfo('EMOTION', em.upper())
            
            # IMDb Url for Drama genre of 
            # movie against emotion Sad 
            if(em == "sad"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'DRAMA......')
                urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
                

            # IMDb Url for Musical genre of 
            # movie against emotion Disgust 
            elif(em == "disgust"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'MUSICAL........')
                
                urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Family genre of 
            # movie against emotion Anger 
            elif(em == "angry"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'FAMILY.......')
                
                urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Thriller genre of 
            # movie against emotion Anticipation 
            elif(em == "neutral"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'THRILLER........')
                
                urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Sport genre of 
            # movie against emotion Fear 
            elif(em == "scared"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'SPORT.........')
                
                urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'

            # IMDb Url for Thriller genre of 
            # movie against emotion Enjoyment 
            elif(em == "happy"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'THRILLER.........')
                
                urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'

            
            

            # IMDb Url for Film_noir genre of 
            # movie against emotion Surprise 
            elif(em == "surprised"):
                root11= tk.Tk()
                root11.withdraw()
                msgbox1=tk.messagebox.showinfo('GENRE ALLOCATED', 'FILM_NOIR.........')
                
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
            rating = soup.find_all("div", {"class": "inline-block ratings-imdb-rating"})
            import tkinter
            from tkinter import ttk

            root12 = tkinter.Tk()
            root12.geometry("600x600")
            root12.title("MOVIES RECOMMENDED FOR DETECTED EMOTION")
            tree = ttk.Treeview(root12)
            tree["columns"]=("one","two")
            tree.column("one", width=200)
            tree.column("two", width=200)
            
            style = ttk.Style(root12)
            style.configure('Treeview', rowheight=45)
                    
            
            tree.heading("one", text="MOVIES")
            tree.heading("two", text="RATINGS")       
            for i in range(9,-1,-1):
                tree.insert("" , 0,    text="", values=(title1[i].text,rating[i].text))
            tree.pack()
            
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
