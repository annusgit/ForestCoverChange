

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
from scipy import ndimage
import numpy as np
import cv2


def convert_labels(gray):
    # 29(0): background/clutter, 76(1): buildings, 150(2): trees,
    # 179(3): cars, 226(4): low_vegetation, 255(5): impervious_surfaces
    # conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    # we are removing the cars class because of excessive downsampling used in the dataset, but not now...
    conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    conversions = {v: k for k, v in conversions.iteritems()}
    new_color = np.zeros(shape=(gray.shape[0], gray.shape[1], 3))
    colors = {0: (255,0,0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 255, 255)}
    for k in conversions.keys():
        new_color[gray == k] = colors[k]
    # print(np.unique(gray))
    return new_color

image = cv2.imread('image1.png')[:620,:380,:]
label = cv2.imread('label1.png')[:620,:380,0]
pred = cv2.imread('pred1.png')[:620,:380,0]
print(label.shape, pred.shape)
print(np.unique(label), np.unique(pred))

# fix pred
pred[pred==0] = 8
pred[pred==1] = 0
pred[pred==8] = 1

pred[pred==3] = 8
pred[pred==4] = 3
pred[pred==8] = 4
pred = ndimage.median_filter(pred, 3)
print((pred==label).sum()/pred.reshape(-1).shape[0])
# np.savetxt('label.txt', label.astype(np.uint8))
# np.savetxt('pred.txt', pred.astype(np.uint8))

pred = convert_labels(pred)
label = convert_labels(label)

pl.subplot(131)
pl.imshow(image)
pl.axis('off')
pl.title('input')
pl.subplot(132)
pl.imshow(label)
pl.axis('off')
pl.title('label')
pl.subplot(133)
pl.imshow(pred)
pl.axis('off')
pl.title('prediction')
pl.show()






