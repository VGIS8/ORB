"""Simple re-implementation of rBRIEF only supporting simple two point test.
"""

import cv2 as cv
import numpy as np
import math

from patterns import bit_pattern_31

def rbrief(features, grey_pyramid):
    """Re-implementation of rBRIEF for use in ORB
    """

    for idx, layer in enumerate(features):
        scale = 1/(idx+1)

        for keypoint in layer:
            # Center is a (y,x) point for use with the image array
            i_center = grey_pyramid[idx][int(keypoint.pt[1] * scale), int(keypoint.pt[0] * scale)]
            theta_trig = (math.cos(keypoint.angle), math.sin(keypoint.angle))

            val = 0
            for test in range(256):
                val |= (get_value(test, 0, theta_trig, i_center) < get_value(test, 1, theta_trig, i_center)) << test
            
            keypoint.class_id = int(val)



def get_value(test, point, theta_trig, i_center):
    p_x = bit_pattern_31[test][point][0]
    p_y = bit_pattern_31[test][point][1]

    ix = round(p_x * theta_trig[0] - p_y * theta_trig[1])
    iy = round(p_x * theta_trig[1] + p_y * theta_trig[0])

    return i_center + iy + ix

pass