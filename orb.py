import cv2 as cv
import numpy as np

from ofast import ofast
from rbrief import rbrief


features, pyramid = ofast('box.png', 500)
features2, pyramid2 = ofast('box_in_scene.png', 500)

ofast_img = cv.drawKeypoints(pyramid[0], features, None, color=(255,0,0))
#cv.imshow('oFAST', ofast_img)
cv.imwrite('test.png', ofast_img)

des1 = rbrief(features, pyramid)
des2 = rbrief(features2, pyramid2)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1[0], des2[0])
matches = sorted(matches, key = lambda x:x.distance)

match_img = cv.drawMatches(pyramid[0],features,pyramid2[0],features2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("matches.png", match_img)
