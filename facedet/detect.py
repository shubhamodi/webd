# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:25:11 2024

@author: shubh
"""

import cv2
import numpy as np
import os
import pickle #to save dataset

#roughly placing of frame
# Get screen dimensions
screen_width = 2000  # Example: replace with actual screen width
screen_height = 780  # Example: replace with actual screen height

# Window dimensions (assumed)
window_width = 1000
window_height = 480

# Calculate position to center the window
x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)

# Create a named window
cv2.namedWindow("frame")

# Move the window to the calculated position
cv2.moveWindow("frame", x_position, y_position)


#using opencv cv2 
video = cv2.VideoCapture(0) #0 means webcamera
get_face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_data=[]
i=0
name=input("Identfied by name : ")
t=0
j=0

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=get_face.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for(x,y,w,h) in faces:
        crop=frame[y:y+h,x:x+w,:]
        resize_image=cv2.resize(crop,dsize=(50,50))
        if(len(face_data)<=100 and i%10==0):      #org_|_
              j+=1
              print(resize_image.shape)
              face_data.append(resize_image)
              cv2.putText(frame,str(j),(50,50),cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(50,50,255),thickness=1)
              cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    #print(k)
    i+=1
    if(j==50):
        break
video.release()
cv2.destroyAllWindows()

#saving faces in pickle
face_data=np.array(face_data)

face_data=face_data.reshape(50,-1)
#finding names
if('names.pkl' not in os.listdir("C:\\Users\\shubh\\Music\\ml_proj\\facedet\\record")):
    names=[name]*50
    print(1)
    with open('record\\names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    print(2)
    with open('record/names.pkl','rb') as f:
        names=pickle.load(f)
    names=names+[name]*50
    with open('record/names.pkl','wb') as f:
        pickle.dump(names,f)

#finding face data in file
if('face_data.pkl' not in os.listdir('record/')):
    print(3)
    with open('record/face_data.pkl','wb') as f:
        pickle.dump(face_data,f)
else:
    print(4)
    obj=[]
    with open('record/face_data.pkl','rb') as f:
        faces=pickle.load(f)
        
    print(faces.shape)
    faces=np.append(faces,face_data,axis=0)
    print(faces.shape)
    #faces.reshape(1,-1)
    with open('record/face_data.pkl','wb') as f:
        pickle.dump(faces,f)









