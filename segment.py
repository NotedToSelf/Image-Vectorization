#Erik Bogeberg
#Watershed segmentation using Python and opencv

import cv2
import numpy as np
src = './coins.jpg'

#read greyscale source image
gray = cv2.imread(src, 0)

#adaptive thresholding
ret, thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

if False:
	cv2.imwrite('./gray.jpg', gray)
	cv2.imwrite('./thresh.jpg', thresh)

#noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

if False:
	cv2.imwrite('./open.jpg', opening)

#sure bg area
sure_bg = cv2.dilate(opening, kernel,iterations=3)

if False:
	cv2.imwrite('./bg.jpg', sure_bg)

#sure fg area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(), 255,0)

#finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

if False:
	cv2.imwrite('./dist.jpg', dist_transform)
	cv2.imwrite('./distthresh.jpg', sure_fg)

if False:
	cv2.imwrite('./unknown.jpg', unknown)

#Marker labeling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0

img = cv2.imread(src, 1)
markers = cv2.watershed(img, markers)

#Watershed marks image edges with -1, replace with red to see edges
copy = img.copy()
copy[markers == -1] = [0,0,0]
copy[markers != -1] = [255,255,255]

copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
print(copy.shape)
cv2.imwrite('./edges.jpg', copy)
#create one dimensional label array
