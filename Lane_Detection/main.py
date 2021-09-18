import os
import sys
import re
import cv2
import argparse
import numpy as np
from time import sleep
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
description='Lane Detection')
parser.add_argument("--video", default = 'LD_video.mp4', help="path to video file")
parser.add_argument("--generate", default = 'No', help="generate an output video")
parser.add_argument("--adjustby", default = 0, help="adjust the position of polygon, e.g. if -10 then shift the polygon to upward, please make sure it is in the range of 0 to 70")
args = parser.parse_args()
adjustby = int(args.adjustby)
#print(adjustby)

if adjustby < 0 or adjustby > (270-160):
    print('polygon adjusting factor is invalid, please make sure it is in the range of 0 to 70')
    sys.exit("aa! errors!")

cap = cv2.VideoCapture(args.video)

counter = 0
img_array = []
fps = cap.get(cv2.CAP_PROP_FPS)
print("input video fps: {}".format(fps))

video_size = (480,270) # set the output video size

while True:
    ret , frame = cap.read()
    counter+=1    
    if frame is None:
        print("End of the video")
        break
    current_frame = cv2.resize(frame, video_size) 
    rows, cols, channels = current_frame.shape
    # create a zero array
    stencil = np.zeros_like(current_frame[:,:,0])

    # specify coordinates of the polygon
    polygon = np.array([[50,270-adjustby], [220,160-adjustby], [360,160-adjustby], [480,270-adjustby]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)
    # cv2.imshow("Video Original" , current_frame)

    # apply polygon as a mask on the frame
    img = cv2.bitwise_and(current_frame[:,:,0], current_frame[:,:,0], mask=stencil)
    # # plot masked frame
    # plt.figure(figsize=(10,10))
    # plt.imshow(img, cmap= "gray")
    # plt.show()

    # apply image thresholding
    ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)
    # # plot image
    # plt.figure(figsize=(10,10))
    # plt.imshow(thresh, cmap= "gray")
    # plt.show()

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
    #print('type of lines: {}'.format(type(lines)))
    
    # create a copy of the original frame
    dmy = current_frame.copy()
    # draw Hough lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

    if (args.generate == "yes"):
        img_array.append(dmy)

    # plot frame
    cv2.imshow("Output",dmy)        

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


# write the sample output video
if (args.generate == "yes"):
    out = cv2.VideoWriter('LD_video_out.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps, video_size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
