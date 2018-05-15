#Image segmentation script.
#Uses skimage for k-means 
import cv2
import numpy as np
from skimage import data, segmentation, color, io, measure

#Create Subpixel Edge Image
#Returns edge map with edges marked with 1, junction points marked with 2  
def get_subpixel(label):
	width, height = label.shape
	edges = np.zeros((width*2-1, height*2-1))
	width, height = edges.shape
	#determine edges from label image
	for x in range(width):
		for y in range(height):
			if ( ((x%2 != 0) and (y%2 == 0)) and label[(x+1)//2][y//2] != label[(x-1)//2][y//2] ):
				edges[x][y] = 1
			if ( ((x%2 == 0) and (y%2 != 0)) and label[x//2][(y+1)//2] != label[x//2][(y-1)//2] ):
				edges[x][y] = 1

	#Fill the gaps between edge pixels
	for x in range(width-1):
		for y in range(height-1):
			if( edges[x+1][y] == 1 and edges[x-1][y] == 1 ):
				edges[x][y] = 1 
			if( edges[x][y+1] == 1 and edges[x][y-1] == 1 ):
				edges[x][y] = 1

	#Find and label junction points
	#This is wrong, need a new solution. Labeling points with 2 will effect further iterations
	for x in range(width-1):
		for y in range(height-1):
			if ( (edges[x+1][y] + edges[x-1][y] + edges[x][y+1] + edges[x][y-1]) > 2 ):
				edges[x][y] = 2
	
	return edges


#main
infile = './images/bird.jpg'
image = io.imread(infile)
height, width, depth = image.shape
#maybe tune these parameters to image size or density??
#Detail fine equation: 
#Detail med equation:
#Detail low equation:

#Get label image
labels = segmentation.slic(image, compactness=20, n_segments=3100)

#Get subpixel edge map
edges = get_subpixel(labels)


#some tests
if False:
	print("original shape: " + str(image.shape))
	print("label map shape: " + str(labels.shape))
	props = measure.regionprops(labels) #label properties
	out = color.label2rgb(labels, image, kind='avg') #color labels with average region color
	io.imsave('output.jpg', out) #write output

