#Image segmentation script.
#Uses skimage for k-means 

from skimage import data, segmentation, color, io
from skimage.future import graph

infile = './images/bird.jpg'

image = io.imread(infile)

#maybe tune these parameters to image size or density??
labels = segmentation.slic(image, compactness=20, n_segments=3100)
out = color.label2rgb(labels, image, kind='avg')
io.imsave('output.jpg', out)

