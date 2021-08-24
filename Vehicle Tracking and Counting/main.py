import cv2
import argparse
import numpy as np
from time import sleep


parser = argparse.ArgumentParser(
description='Vehicle Tracking and Counting')
parser.add_argument("--video", default = 'video.mp4', help="path to video file")
args = parser.parse_args()

min_width=80 #Minimum width of the rectangle
min_height=80 #Minimum height of the rectangle

offset=6 #Allowable error between pixel, default=6   

line_ypos=550 #position of line 

delay= 60 #Video FPS

vehicles_in= 0
vehicles_out= 0
detec = [] 
	
def roi_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture(args.video)
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , current_frame = cap.read()
    if current_frame is None:
        print("End of the video")
        break
    rows, cols, channels = current_frame.shape
    # line_ypos=int(0.7638*rows)

    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    dilate = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(current_frame, (0, line_ypos), (int(cols/2), line_ypos), (255,127,0), 3)
    cv2.line(current_frame, (int(cols/2), line_ypos), (cols, line_ypos), (127,255,0), 3)
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(current_frame,(x,y),(x+w,y+h),(0,255,0),2)        
        center = roi_center(x, y, w, h)
        detec.append(center)
        cv2.circle(current_frame, center, 4, (0, 0,255), -1)

        for (x,y) in detec:  # For each center of particular detected regions in the current frame
            if y<(line_ypos+offset) and y>(line_ypos-offset): # If the center lies on the line
                if x < int(cols/2):
                    vehicles_in+=1
                    print("Vehicle in : "+str(vehicles_in)) 
                    cv2.line(current_frame, (0, line_ypos), (int(cols/2), line_ypos), (0,0,255), 3) 
                elif x > int(cols/2):
                    vehicles_out+=1
                    print("Vehicle out : "+str(vehicles_out)) 
                    cv2.line(current_frame, (int(cols/2), line_ypos), (cols, line_ypos), (0,0,255), 3)
                else:
                    pass 
                detec.remove((x,y))
                      
       
    cv2.putText(current_frame, "VEHICLE ENTER COUNT : "+str(vehicles_in), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 50),2)
    cv2.putText(current_frame, "VEHICLE EXIT COUNT : "+str(vehicles_out), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 50),2)
    cv2.putText(current_frame, "TOTAL VEHICLE COUNT : "+str(vehicles_out+vehicles_in), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 50),2)
    cv2.imshow("Video Original" , current_frame)
    cv2.imshow("Detector",dilate)

    if cv2.waitKey(1) == 27:
        break
    # key = cv2.waitKey(0)
    # while key not in [ord('q'), ord('k')]:
    #     key = cv2.waitKey(0)
    # # Quit when 'q' is pressed
    # if key == ord('q'):
    #     break
cv2.destroyAllWindows()
cap.release()
