import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'image'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]      #finding all measurement in database images
        encodelist.append(encode)
    return encodelist

def markattendances(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])     # finding the FACE of the image
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')    # upadting in excel sheet
            f.writelines(f'\n{name},{dtString}')


encodelistknown = findEncodings(images) # completion of getting measuretment of the face
print('encoding complete')

cap = cv2.VideoCapture(0)       #initaling the CAMERA

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown,encodeFace)   # matching the face
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)                  # forming rectangle around face and gettimg measurement to match
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,225,225),2)
            markattendances(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)



#faceloc = face_recognition.face_locations(imgelon)[0]
#encodeelon = face_recognition.face_encodings(imgelon)[0]
#cv2.rectangle(imgelon,(faceloc[3],faceloc[0],faceloc[2],faceloc[1]),(255,0,255),2)

#faceloctest= face_recognition.face_locations(imgtest)[0]
#encodetest = face_recognition.face_encodings(imgtest)[0]
#cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0],faceloctest[1],faceloctest[2]),(255,0,255),2)

#results = face_recognition.compare_faces([encodeelon],encodetest)
#facedis = face_recognition.face_distance([encodeelon],encodetest)

#imgelon = face_recognition.load_image_file('image/elontest.jpg')
#imgelon =cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
#imgtest = face_recognition.load_image_file('image/elon.jpg')
#imgtest =cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)