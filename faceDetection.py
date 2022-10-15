import cv2
from cv2 import imshow
import numpy as np
import time

img=cv2.imread("im.jpg")

resized=cv2.resize(img,(450,450),interpolation=cv2.INTER_AREA)


gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
cv2.imshow("human face",resized)

blur=cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)

haar_cascade= cv2.CascadeClassifier("face.xml")
face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
print("number of faces found = "+str(len(face_rect)))

for x,y,w,h in face_rect:
    new=cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("detected faces",new)

cv2.waitKey(0)