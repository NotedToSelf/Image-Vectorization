#Erik Bogeberg
#Image to SVG conversion

import cv2
import numpy as np

#Image Segmentation






#main

#read input
original = cv2.imread(filename='./colors.png', flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0
#Create a segmentation label image of size (2*width - 1, 2*height - 1)
width, height, depth = original.shape
segmented = np.zeros((2*width-1, 2*height-1))


#cv2.imwrite(filename='./output.png', img=(original * 255.0).clip(0.0, 255.0).astype(np.uint8))



def write_labels(original, labels):
	
