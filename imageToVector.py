#Erik Bogeberg
#Image to SVG conversion

import cv2
import numpy as np

#Writes labels based on image segments
def write_labels(contours, labels):
	current = 1;
	for i in contours:
		x,y,w,h = cv2.boundingRect(i)
		labels[x:x+w][y:y+h] = current
		current += 1

#Draws edges based on labeled segments
def draw_edges(labels, edges):
	rows, cols = edges.shape
	for x in range(rows):
		for y in range(cols):
			if x % 2 == 1 and y % 2 == 0:
				if labels[int((x+1)/2)][int(y/2)] != labels[int((x-1)/2)][int(y/2)]:
					edges[x][y] = 0;
			elif x % 2 == 0 and y % 2 == 1:
				if labels[int(x/2)][int((y+1)/2)] != labels[int(x/2)][int((y-1)/2)]:
					edges[x][y] = 0;
			else:
				edges[x][y] = 255;



#main

#read input
original = cv2.imread(filename='./colors.png', flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0
width, height, depth = original.shape

#Canny edge detection of grayscale image to obtain image contours
gray = cv2.imread('./colors.png', 0);
gray = cv2.Canny(gray, 0, 255);
im, contours, heirarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[1:]

#Create a label map of original image segments
labels = np.zeros((width, height))
write_labels(contours, labels)

#Draw an endge map based on found labels
edges = np.full((2*width-1, 2*height-1), 255)
draw_edges(labels, edges)

cv2.imwrite('./output.png', edges)


	
