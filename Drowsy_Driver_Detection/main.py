import numpy as np
import cv2
import _thread as thread
import subprocess
import sys
#import winsound # For Ms Windows
# import playsound # For Linux, if you got error, please install it using 'pip install pygobject playsound'
from playsound import playsound

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def beep():
  for i in xrange(2):
    playsound("beep.mp3")
    #winsound.Beep(1500, 250)

cap = cv2.VideoCapture(0)

count = 0
iters = 0
while(True):
      ret, cur = cap.read()
      gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors=1, minSize=(10,10))
      for (x,y,w,h) in faces:
      	#cv2.rectangle(cur,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = cur[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
          print ("Eyes closed")
        else:
          print ("Eyes open")
        count += len(eyes)
        iters += 1
        if iters == 9:
          iters = 0
          if count == 0:
            print ("Drowsiness Detected!!!")
            cv2.putText(cur, "Drowsiness Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150),2)
            thread.start_new_thread(beep,())
          count = 0
        for (ex,ey,ew,eh) in eyes:
        	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
      cv2.imshow('frame', cur)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
cap.release()