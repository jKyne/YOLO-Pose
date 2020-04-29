# -*- coding: UTF-8 -*-
import os
import cv2
import time
 
def picvideo(path,size):
    filelist = os.listdir(path) 
    filelist.sort(key=lambda x:int(x[-9:-5]))
    fps = 30
    file_path = str(int(time.time())) + ".mp4"
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') 
 
    video = cv2.VideoWriter( file_path, fourcc, fps, size )
 
    for item in filelist:
        if item.endswith('.jpeg'): 
            item = path + '/' + item 
            img = cv2.imread(item)  
            video.write(img)  
 
    video.release() 

picvideo("data/Results",(768,432))
 
