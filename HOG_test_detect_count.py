#HOG_test.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:40:31 2016

@author: azaterka
"""
import cv2
import imutils
import datetime

camera = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

detectionCount = 0
detectionSum = 0
maxDetectionCount = 50
isInteresting = False

while True:
    start = datetime.datetime.now()
    # grab the current frame
    grabbed, frame = camera.read()

    if not grabbed:
        print('Problem reading frame')
        break
#    frame_original = frame.copy()
#    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    
    pad = 16
    win = 8
    scale = float(1.05)
    (rects, weights) = hog.detectMultiScale(frame, winStride=(win,win),padding=(pad,pad), scale=scale, useMeanshiftGrouping=0)
#    print(rects)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    detectionCount += 1
    detectionSum += (len(rects) > 0)
    if detectionCount > maxDetectionCount:
         isInteresting = float(detectionSum)/maxDetectionCount > .5
         detectionCount = 0
         detectionSum = 0
         
            
    
    cv2.imshow("Detections", frame)
    print("Frame rate:",str(1/(datetime.datetime.now() - start).total_seconds())[:5],"fps",detectionCount,detectionSum,isInteresting,(float(detectionSum)/maxDetectionCount))
    
    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'):
        break
#
#frame = cv2.imread("person1.jpg")
#frame = imutils.resize(frame, width=min(400, frame.shape[1]))

#pad = 64
#(rects, weights) = hog.detectMultiScale(frame, winStride=(8,8),padding=(pad,pad), scale=float(1.5), useMeanshiftGrouping=0)
#print(rects)
#for (x, y, w, h) in rects:
#    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#cv2.imshow("Detections", frame)
#cv2.waitKey(0)
cv2.destroyAllWindows()