import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('image/sau.jpg')
imgelon =cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('image/sau2.jpg')
imgtest =cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceloc[3],faceloc[0],faceloc[2],faceloc[1]),(255,0,255),2)

faceloctest= face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0],faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeelon],encodetest)
facedis = face_recognition.face_distance([encodeelon],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('original',imgelon)
cv2.imshow('test',imgtest)
cv2.waitKey(0)