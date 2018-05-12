#Erik Bogeberg
#Image to SVG conversion

import cv2
import numpy as np


src = './coins.jpg'
edge = './edges.jpg'


#main
edges = cv2.imread(edge, 1)

cv2.imwrite(edge, edges)
		
