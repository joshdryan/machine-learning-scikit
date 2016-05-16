# isolatedigits.py
# Josh Ryan

# import matplotlib
from matplotlib.path import Path

# import Scikit
from skimage.transform import resize
from skimage.filters import threshold_adaptive

# import SciPy stack
import scipy.ndimage as ndimage
import numpy as np
	
class IsolateDigit():

	def __init__(self):
		self._min_x = 0
		self._min_y = 0
		self._max_x = 0
		self._max_y = 0

	def get_loc(self):
		return (self._min_x, self._min_y)

	def adaptivethreshold(self, image):
	    '''Uses an adaptive threshold to change image to binary. This is done using 
	    the Adaptive Thresholding technique, begins by using Gaussian filtering to 
	    remove noise, follows by creating a binary image with Otsu's Method. 
	    Reference: https://en.wikipedia.org/wiki/Otsus_method'''

	    # Image is thresholded, inverted, dilated, and has holes filled. 
	    thresh = threshold_adaptive(image, 41, offset=10)
	    thresh = np.invert(thresh)
	    thresh = ndimage.grey_dilation(thresh, size=(2,2))
	    return ndimage.binary_fill_holes(thresh)

	def bounding_box(self, iterable):
	    '''Returns bounding box vertices around each contour'''
	    self._min_x = np.min(iterable[:,1], axis=0)
	    self._max_x = np.max(iterable[:,1], axis=0)
	    self._min_y = np.min(iterable[:,0], axis=0)
	    self._max_y = np.max(iterable[:,0], axis=0)
	    v = np.array([self._min_x, self._min_y, self._max_x, self._max_y])

	    # use vertices to create single numpy array
	    x = np.array([v[0], v[0], v[2], v[2], v[0]])
	    y = np.array([v[3], v[1], v[1], v[3], v[3]])
	    xycrop = np.vstack((x, y)).T

	    # Create Path object from vertices
	    return Path(xycrop, closed=False)

	def finddigit(self, image, contour, check=False):
	    '''Finds each digit in image, crops image'''
	    # find path
	    pth = self.bounding_box(contour)

	    #Find all points inside image
	    nr, nc = image.shape
	    ygrid, xgrid = np.mgrid[:nr, :nc]
	    xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

	    # Find and reshape mask region
	    mask = pth.contains_points(xypix)
	    mask = mask.reshape(image.shape)

	    # find image contained in masked region
	    masked = np.ma.masked_array(image, ~mask)

	    if check == True:
	    	return masked
	    else:
	    	return masked[self._min_y:self._max_y, self._min_x:self._max_x]

	def formatdigit(self, image, contour):
		# find larger edge
		edgex = float((self._max_x - self._min_x)) / 2
		edgey = float((self._max_y - self._min_y)) / 2

		# Create edge border around digit
		if (self._max_x - self._min_x) > (self._max_y - self._min_y):
			edge = edgex + float(edgex)/15
		else:
			edge = edgey + float(edgey)/15

		# find center point
		xc = self._min_x + edgex
		yc = self._min_y + edgey

		masked = self.finddigit(image, contour, check=True)
		masked = masked[yc-edge:yc + edge, xc - edge:xc + edge]
		masked = resize(masked, (28, 28))
		return ndimage.grey_dilation(masked, size=(1,1))