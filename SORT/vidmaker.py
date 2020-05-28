# import cv
# import numpy as np
# import os
# from os.path import isfile, join
# pathIn= './output/'
# pathOut = 'video.avi'
# fps = 25
# frame_array = []
# files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# #for sorting the file names properly
# files.sort(key = lambda x: x[5:-4])
# files.sort()
# frame_array = []
# files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# #for sorting the file names properly
# files.sort(key = lambda x: x[5:-4])
# for i in range(len(files)):
#     filename=pathIn + files[i]
#     #reading each files
#     img = cv.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
    
#     #inserting the frames into an image array
#     frame_array.append(img)
# out = cv.VideoWriter(pathOut,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
# for i in range(len(frame_array)):
#     # writing to a image array
#     out.write(frame_array[i])
# out.release()         ./data/mobilenetskipadd_cross_entropy_augmented_rgb/000000_bundle.jpg


import cv2 as cv
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort()#key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)-1):
        filename=pathIn + files[i]
        #reading each files
        img = cv.imread(filename)
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv.VideoWriter(pathOut,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= './output/'
    pathOut = 'video_seg_test_hd2.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()