#!/usr/bin/env python
#
# This example shows the diffent forms of plotting for xSVMC contextualized binary classifications
# of one of the classes of handwritten digits.
# 
# The example is a modified version of the OpenCV tutorial entitled 'OCR of Hand-written Data using SVM' 
# posted on 
#       https://docs.opencv.org/4.5.3/dd/d3b/tutorial_py_svm_opencv.html
#
# Keywords: xSVMC, binary classification, contextualized classification, plotting
#


import cv2 as cv
import numpy as np
import os, sys

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    print('inserting {!r} into sys.path'.format(_path_to_lib_))
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xsvmc import xSVMC
from xsvmlib.xploting import BuoyancyPlot

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

digits = cv.imread(cv.samples.findFile('examples/digits.png'),0)
if digits is None:
    raise Exception("we need the digits.png image from samples/data here !")

cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

digits_coord = [[i,j] for i in range(50) for j in range(50)]

deskewed = [list(map(deskew,row)) for row in train_cells]
hog_data = [list(map(hog,row)) for row in deskewed]

X_train = np.float32(hog_data).reshape(-1,64)
y_train = np.repeat(np.arange(10),250)[:,np.newaxis]

deskewed = [list(map(deskew,row)) for row in test_cells]
hog_data = [list(map(hog,row)) for row in deskewed]

X_test = np.float32(hog_data).reshape(-1,bin_n*4)
y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

degree = 2.0
gamma = 5.383
coef0 = 0.0
c0 =  2.67
k = 1

clf = xSVMC(kernel='poly', C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
clf.fit(X_train, y_train.ravel())

predictions = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Create a complete list of pictures for both train and test sets 
train_pics = []
for coord in range(len(digits_coord)):
    train_pics.append(deskew(train_cells[digits_coord[coord][0]][digits_coord[coord][1]]))

test_pics = []
for coord in range(len(digits_coord)):
    test_pics.append(deskew(test_cells[digits_coord[coord][0]][digits_coord[coord][1]]))

ref_SVs = clf.support_

mu_values = []
nu_values = []
buoyancy_values = []
object_pics = []
misvPro_pics = []
misvCon_pics = []
labels = []

ifs = clf.is_member_of(X_test[:6], 0)
test_pics = test_pics[:6]

for i in range(len(ifs)):
    idxPositiveMISV = ref_SVs[ifs[i].mu_hat.misv_idx]
    idxNegativeMISV = ref_SVs[ifs[i].nu_hat.misv_idx]

    misvPro_pics.append(train_pics[idxPositiveMISV])
    misvCon_pics.append(train_pics[idxNegativeMISV])

buoy_chart = BuoyancyPlot(ifs, 0, test_pics, misvPro_pics, misvCon_pics, zoom=2)
buoy_chart.plot(title="Digit classification - Mu-nu values", version="mu_nu_values", show_legend=False, samples_per_page=4)
buoy_chart.plot(title="Digit classification - Buoyancy values", version="buoyancy_values", show_legend=False, samples_per_page=4)