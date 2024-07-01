# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:02:38 2024

@author: shubh
"""

import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from win32com.client import Dispatch
def say(str_):
    speak=Dispatch("SAPI.SpVoice")
    speak.Speak(str_)


video = cv2.VideoCapture(0) #0 means webcamera
get_face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#importing recorded data
with open('record/names.pkl','rb') as w:
    LABELS=pickle.load(w)

obj=[]
with open('record/face_data.pkl','rb') as f:
    FACES=pickle.load(f)
    #while True:
        #try:
         #   obj.append(pickle.load(f))

        #except EOFError:
        #    break
#FACES=np.array(obj)
print(FACES.shape)
#FACES=FACES.reshape(1,-1)
print(len(LABELS))
# Training based on data
#if len(LABELS) == len(FACES) and LABELS and FACES:
    # Training based on data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)
#else:
 #   print(f"Data mismatch: {len(FACES)} face samples and {len(LABELS)} labels found.")

attendance_folder = "Attendance"
#os.makedirs(attendance_folder, exist_ok=True)


imgbackground=cv2.imread("annie-spratt-wuc-KEIBrdE-unsplash.jpg")
columns=['NAME',"TIME"]
face_data=[]
i=0
j=0
t=0
while True:
    face_data=[]
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=get_face.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    
    for(x,y,w,h) in faces:
        crop=frame[y:y+h,x:x+w,:]
        resize_image=cv2.resize(crop,dsize=(50,50)).flatten().reshape(1,-1)
        print(resize_image.shape)
        #resize_image=resize_image.reshape(100,-1)
        print(resize_image.shape)
        
      #  if(len(face_data)<=100 and i%10==0):      #org_|_
      #      j+=1
       #     if(t==0):
        #      #t+=1
         #     face_data.append(resize_image)
       # face_data1=np.array(face_data)
        #face_data1=face_data1.reshape(100,-1)
        output=knn.predict(resize_image)
        #output.reshape(1,-1)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        path=os.path.join(attendance_folder, "Attendance_"+date+".csv")
        print(path)
        exist=os.path.isfile(path)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=())
        attendance=[str(output[0]),str(timestamp)]
        imgbackground[162:162+480,55:55+640]=frame
        
        cv2.imshow("frame",imgbackground)
        k=cv2.waitKey(1)
        print(k)
        print(ord('o'))
        print(ord('q'))
        if(k==ord('o')):
            say("Amazing!!!!!"+str(output[0])+", is present")
            time.sleep(5)
            print('waah')
            print(exist)
            if exist:
                with open(os.path.join(attendance_folder, f"Attendance_"+date+".csv"),"+a",newline='') as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                print("naah")
                with open(os.path.join(attendance_folder, f"Attendance_"+date+".csv"),'w',newline='') as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerow(attendance)
                
        if(k==ord('p')):
           say("Have a Good Day !")
           t=1
           break
    if(t):
           break
video.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        

        