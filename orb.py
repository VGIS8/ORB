import cv2 as cv

from ofast import ofast
from rbrief import rbrief

features, pyramid = ofast('elhest.jpg', 500)

grey_pyramid = []
for layer in pyramid:
    grey_pyramid.append(cv.cvtColor(layer, cv.COLOR_BGR2GRAY))

descriptors = rbrief(features, grey_pyramid)
ofast_img = cv.drawKeypoints(pyramid[0], features[0], None, color=(255,0,0))
cv.imwrite('best_kp.png', ofast_img)