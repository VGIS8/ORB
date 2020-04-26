"""Simple re-implementation of rBRIEF only supporting simple two point test.
"""

import cv2 as cv
import numpy as np
import math

from patterns import bit_pattern_31

def rbrief(features, grey_pyramid):
    """Re-implementation of rBRIEF for use in ORB
    """
    
    descriptors = np.empty((len(features), len(features[0]), 32), 'uint8')

    for L_idx, layer in enumerate(features):
        scale = 1/(L_idx+1)

        for K_idx, keypoint in enumerate(layer):
            # Center is a (y,x) point for use with the image array
            i_center = grey_pyramid[L_idx][int(keypoint.pt[1] * scale), int(keypoint.pt[0] * scale)]
            theta_trig = (math.cos(keypoint.angle), math.sin(keypoint.angle))

            val = 0

            for byte in range(32):
                for bit in range(8):
                    val |= (get_value(bit*byte, 0, theta_trig, i_center) < get_value(bit*byte, 1, theta_trig, i_center)) << bit
                
                descriptors[L_idx][K_idx][byte] = val

    return descriptors


def get_value(test, point, theta_trig, i_center):
    p_x = bit_pattern_31[test][point][0]
    p_y = bit_pattern_31[test][point][1]

    ix = round(p_x * theta_trig[0] - p_y * theta_trig[1])
    iy = round(p_x * theta_trig[1] + p_y * theta_trig[0])

    return i_center + iy + ix


def toOpenCVMatch(descriptors, size):
    """Will convert the feature descriptors of *size* bits in a layer to an array of bytes
    to work with OpenCVs matching functions

    Args:
        descriptors (int): an integer representation of a bitstring
        size (int): Size of the descriptors in bytes
    """

    out = np.empty((len(features),32), 'uint8')

    for idx, feature in enumerate(features):
        x = feature.class_id
        for i in range(size-1, -1, -1):
            out[idx][i] = (x & 0xFF)
            x = x >> 8
    
    return out