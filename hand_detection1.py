import cv2
import numpy as np
from sklearn.metrics import pairwise

cap = cv2.VideoCapture(0)
kernelOpen = np.ones((5,5))#if jiggers are present other than yellow area
kernelClose = np.ones((20,20)) #if jiggers are present in yellow area

#HSV color range which should be detected
lb = np.array([20,100,100])
ub = np.array([120,255,255])

while True:
    ret, frame = cap.read()
    flipped = cv2.flip(frame, 1)
    flipped = cv2.resize(flipped,(500,400))

    #use HSV of yellow to detect only yellow color
    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imgSegFlipped = cv2.flip(imgSeg, 1)
    imgSegFlipped = cv2.resize(imgSegFlipped,(500,400))
    
    #masking and filtering all shades of yellow
    mask = cv2.inRange(imgSegFlipped, lb, ub)
    mask = cv2.resize(mask,(500,400))

    #apply morphology to avoid jiggers
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskOpen = cv2.resize(maskOpen,(500,400))
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    maskClose = cv2.resize(maskClose,(500,400))
    
    final = maskClose
    _, conts, h = cv2.findContours(maskClose,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    
    if(len(conts)!=0): #draws the contours of that object which has the highest
        b = max(conts, key=cv2.contourArea)
        west = tuple(b[b[:, :, 0].argmin()][0]) #gives the co-ordinate of the left extreme of contour
        east = tuple(b[b[:, :, 0].argmax()][0]) #above for east i.e right
        north = tuple(b[b[:, :, 1].argmin()][0])
        south = tuple(b[b[:, :, 1].argmax()][0])
        centre_x = (west[0]+east[0])/2
        centre_y = (north[0]+south[0])/2

        cv2.drawContours(flipped, b, -1, (0,255,0), 3)
        cv2.circle(flipped, west, 6, (0,0,255), -1)
        cv2.circle(flipped, east, 6, (0,0,255), -1)
        cv2.circle(flipped, north, 6, (0,0,255), -1)
        cv2.circle(flipped, south, 6, (0,0,255), -1)
        cv2.circle(flipped, (int(centre_x),int(centre_y)), 6, (255,0,0), -1)#plots centre of the area
    
    cv2.imshow('video', flipped)
    #cv2.imshow('mask', mask)
    #cv2.imshow('mask open', maskOpen)
    #cv2.imshow('mask close', maskClose)
    if cv2.waitKey(1) & 0xFF == ord(' '):#exiting
        break

cap.release()
cv2.destroyAllWindows()
