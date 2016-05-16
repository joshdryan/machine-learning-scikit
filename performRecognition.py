# ================================
# Recognizing hand-written digits
# By Josh Ryan
# ================================

# A program showing how scikit and matplotlib can be used to recognize, 
# plot, and convert images of hand-written digits to strings. 

# This example is based on the following tutorial:
# http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and
# -python.html

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# import Scikit
from sklearn.externals import joblib
from skimage.feature import hog
from skimage import measure
from skimage.transform import resize

# import SciPy stack
import scipy.ndimage as ndimage
import numpy as np

from isolatedigits import IsolateDigit

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def scale(img):
    scale = img.shape
    #img = img[img.shape[0], img.shape[1]]
    scalenum = float(840/scale[0])
    scaled = float(scale[1]*scalenum)
    scaled = resize(img, (840, int(scaled)), preserve_range=True)
    return scaled

if __name__=='__main__':
    # Load the classifier
    clf = joblib.load("digits_cls.pkl")

    # Read the input image, convert to grayscale
    im = ndimage.imread('photo_2.jpg', flatten=True)

    # Set subplots
    ax1 = plt.subplot(1, 2, 1, adjustable='box-forced')
    ax2 = plt.subplot(1, 2, 2, adjustable='box-forced')
    ax1.set_title("Input")
    ax2.set_title("Identified Digits")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Rescale image, Apply adaptive Threshold
    scaled = scale(im)
    digit = IsolateDigit()
    thresh = digit.adaptivethreshold(im)
    thresh = scale(thresh)

    # Find contours in image
    contours = measure.find_contours(thresh, .1)
    # Remove contours smaller than 50 in the Y direction
    n = 0
    for x,contour in reversed(list(enumerate(contours))):
        diameter = np.max(contour[:,0].max() - contour[:,0].min())
        if diameter<50: # threshold to be refined for your actual dimensions!
            n +=1
            removearray(contours, contour)

    # isolate and Identify digits
    for contour in contours:
        # Plot Bounding rectangles
        pth = digit.bounding_box(contour)
        patch = patches.PathPatch(pth, fill=False, lw=2, ec='r')
        ax2.add_patch(patch)

        # Set trimmed image to proper square dimensions
        dim1 = digit.formatdigit(thresh, contour)

        # create hog vector, use vector to predict digits
        hog_fd = hog(dim1, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        prediction = clf.predict(np.array([hog_fd], 'float64'))
        ax2.text(digit.get_loc()[0], digit.get_loc()[1], (str(int(prediction[0]))), color='b')

    # Plot Images
    ax1.imshow(im, cmap=plt.cm.gray)
    ax2.imshow(scaled, cmap=plt.cm.gray)
    plt.show()