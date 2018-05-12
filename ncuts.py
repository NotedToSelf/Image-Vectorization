#Attemp to implement "Normalized Cuts and Image Segmentation"

from skimage import data, segmentation, color, io
from skimage.future import graph

infile = 'bird.jpg'

image = io.imread(infile)

#maybe tune these parameters to image size or density??
labels = segmentation.slic(image, compactness=20, n_segments=3100)
out = color.label2rgb(labels, image, kind='avg')
io.imsave('output.jpg', out)

