import sys
import numpy as np
from skimage import segmentation, color, io
import random
import struct
import cv2


def rgb2hex(rgb):
    return struct.pack('BBB', *rgb).encode('hex')


# Modification of opencv label2rbg_avg function. Labels regions with random colors
def label2rgb_avg(label_field, image):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    for label in labels:
        mask = (label_field == label).nonzero()
        col = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
        out[mask] = col
    return out


def precedence(contour, cols):
    tolerance = 20
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance) * tolerance) * cols+ origin[0]


# Main
name = sys.argv[1]
infile = './images/' + name
image = io.imread(infile)

# roll color channels RBG --> BRG
image = image[..., ::-1]

# If gray-scale, convert to 3 channel gray-scale
if len(image.shape) == 2:
    image = np.stack((image,) * 3, -1)

height, width, depth = image.shape
size = height * width
print('\ntotal pixels:' + str(size))

# sparse detail
if sys.argv[2] == 'sparse':
    labels = segmentation.slic(image, compactness=1, n_segments=50, enforce_connectivity=True)
# Low detail
elif sys.argv[2] == 'low':
    labels = segmentation.slic(image, compactness=5, n_segments=800, enforce_connectivity=True)
# medium detail
elif sys.argv[2] == 'med':
    labels = segmentation.slic(image, compactness=10, n_segments=2000, enforce_connectivity=True)
# high detail
elif sys.argv[2] == 'high':
    labels = segmentation.slic(image, compactness=20, n_segments=3500, enforce_connectivity=True)
# ultra detail
elif sys.argv[2] == 'ultra':
    labels = segmentation.slic(image, compactness=30, n_segments=7000, enforce_connectivity=True)
else:
    print('Invalid Detail Argument.')

# Labels to RBG, assigning average color of region
preview = color.label2rgb(labels, image, kind='avg')
cv2.imwrite('./labelPreview.jpg', preview)

# Color labels with random colors
out = label2rgb_avg(labels, image)
cv2.imwrite('./randomColors.jpg', out)

# Gray-scale version
grayEdge = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./randomGray.jpg', grayEdge)

# Canny edge detection
canny = cv2.Canny(grayEdge, 0, 255)

# Copy() returns the image format expected by findContours used later
preview = preview.copy()

# Ensure edge image is binary
ret, canny = cv2.threshold(canny, 127, 255, 0)
cv2.imwrite('./Canny.jpg', canny)

# Dilate and erode to fill gaps in canny edges
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
dia = cv2.dilate(canny, kernel3, iterations=2)
edges = cv2.erode(dia, kernel5, iterations=1)
cv2.imwrite('./filled.jpg', edges)

# Dilate and erode to fill gaps in canny edges
# Get all contours in canny edge image
# Use CHAIN_APPROX_NONE to store all contour points. We need more than 2 points to draw cubic benzer curves in our svg
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours.sort(key=lambda x:precedence(x, im2.shape[1]))

# Draw contours onto color preview
mapped = cv2.drawContours(preview, contours, -1, (0, 0, 0), 1)
height, width, _ = mapped.shape
cv2.imwrite('./mapped.jpg', mapped)

# Write contours to svg
f = open('./output.svg', 'w+')
# f.write('<?xml version="1.0" encoding="utf-8"?>')
f.write('<svg width="' + str(width) + '" height="' + str(height) + '" xmlns="http://www.w3.org/2000/svg" version="1.1">')
for cont in contours:
    c = cont
    # Calculate color of contour
    M = cv2.moments(c)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        r, g, b = preview[cY][cX]
    else:
        b = 0
        g = 0
        r = 0
    val = rgb2hex((b, g, r))
    x, y = cont[0][0]
    f.write('\n<polygon points="')
    for i in range(len(c)):
        x, y = c[i][0]
        f.write(str(x) + ',' + str(y) + ' ')
    f.write('" stroke="#' + str(val) + '" fill="#' + str(val) + '"/>')
f.write('</svg>')
f.close()
