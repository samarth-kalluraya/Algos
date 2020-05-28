import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 

# Minimum size of a segmention to be considered an object
min_bbox_area = 50 # (pixels)

# Input image size 
img_size = [512, 640]
# We are given a bundle image which consists of input image, ground truth + prediction
input_bundle_file = "../data/fla_detection/000034_bundle.jpg"
bundle_img =  cv.imread(input_bundle_file)

# Input image
img  = bundle_img[:,:img_size[1]]
print(bundle_img.shape)
if bundle_img.shape[1] == 3*img_size[1]:
    seg = bundle_img[:,2*img_size[1]:]

# Find the binary mask (note that segmentation has blue color
mask = seg[:,:,0] > 200
mask = 255*mask.astype(np.uint8)

# Find contours 
import pdb
pdb.set_trace()

_, contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

bbox_list = []

for cnt in contours:
    rect = cv.boundingRect(cnt)
    x, y, w, h = rect 
    if w*h > min_bbox_area:
        bbox_list.append([x, y, w, h])

for box in bbox_list:
	start_point = tuple(box[0:2])
	end_point   = (start_point[0] + box[2], start_point[1] + box[3])
	cv.rectangle(img, start_point, end_point, color=(255, 30, 50))
plt.imshow(img)
plt.show()
print(bbox_list)
