from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import cv2
import pickle
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/hfd.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)



while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w : ]
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1, -1)
        output  = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendace"+date+".csv")
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
        attendance = [str(output[0]), str(timestamp)]
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('o'):
        if exist:
            with open("Attendance/Attendance"+date+".csv", "+a") as cf:
                writer = csv.writer(cf)
                writer.writerow(["Name", "Time"])
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance"+date+".csv", "+a") as cf:
                writer = csv.writer(cf)
                writer.writerow(["Name", "Time"])
                writer.writerow(attendance)
                
                
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
