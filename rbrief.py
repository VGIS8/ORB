"""Simple re-implementation of rBRIEF only supporting simple two point test.
"""

import cv2 as cv
import numpy as np
import math

from patterns import bit_pattern_31

max_ix_iy = [0,0]

def rbrief(features, grey_pyramid):
    global max_ix_iy
    """Re-implementation of rBRIEF for use in ORB
    """
    
    descriptors = np.empty((len(features), 32), 'uint8')


    for K_idx, keypoint in enumerate(features):
        scale = 1.0/(keypoint.octave+1)
        # Center is a (y,x) point for use with the image array
        center = [int(keypoint.pt[1] * scale), int(keypoint.pt[0] * scale)]

        theta_trig = (math.cos(keypoint.angle), math.sin(keypoint.angle))

        val = 0

        for byte in range(32):
            for bit in range(8):
                val |= (
                    get_value(bit*byte, 0, theta_trig, center, grey_pyramid[keypoint.octave])
                    < 
                    get_value(bit*byte, 1, theta_trig, center, grey_pyramid[keypoint.octave])) << bit
            
            descriptors[K_idx][byte] = val
    print(max_ix_iy)
    return descriptors


def get_value(test, point, theta_trig, center, img):
    global fuck
    global max_ix_iy
    # Get the non rotated test points
    p_x = bit_pattern_31[test][point][0]
    p_y = bit_pattern_31[test][point][1]

    # Get the rotation offset
    ix = round(p_x * theta_trig[0] - p_y * theta_trig[1])
    iy = round(p_x * theta_trig[1] + p_y * theta_trig[0])

    if ix > max_ix_iy[0]:
        max_ix_iy[0] = ix
    if iy > max_ix_iy[1]:
        max_ix_iy[1] = iy

    try:
        return img[center[0] + iy, center[1] + ix]
    except IndexError:
        print(f"ix,iy: ({ix},{iy})")
        return 0
