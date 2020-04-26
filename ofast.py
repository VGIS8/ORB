import cv2 as cv
import numpy as np
import math


def get_pyramid(image, py_image):
    """Generate the scale pyramid for the image
    """

    py_image.append(image)
    py_levels = 3

    for level in range(py_levels):
        image = cv.pyrDown(image)
        py_image.append(image)


def umax(half_patch_size):
    """Get a list of line width at different heights of the circle
    """
    u_max = []
    vmax = math.floor(half_patch_size * math.sqrt(2.00)/2 + 1)
    vmin = math.ceil(half_patch_size * math.sqrt(2.00)/2)
    for v in range(0, vmax + 1):
        u_max.append(round(math.sqrt(float(half_patch_size) * half_patch_size - v*v)))
    
    v0 = 0
    for v in range(half_patch_size, vmin - 1, -1):
        while (u_max[v0] == u_max[v0 +1]):
            v0 = v0 + 1
        u_max[v] = v0
        v0 = v0 + 1
    
    return u_max


def ICAngles(image, keypoints, half_patch_size, u_max):
    """Calculate angles via Intensity Centroids
    """
    
    kp_position = cv.KeyPoint_convert(keypoints)
    ptsize = len(kp_position)
    
    for ptidx in range(ptsize):
        
        m_01 = 0
        m_10 = 0

        for u in range(-half_patch_size, half_patch_size+1):
            m_10 = m_10 + u * image[int(kp_position[ptidx,1]),int(kp_position[ptidx,0])+u]
        
        for v in range(1,half_patch_size+1):
            v_sum = 0
            d = u_max[v]
            for u in range(-d, d-1):
                val_plus = int(image[int(kp_position[ptidx,1])+v, int(kp_position[ptidx,0])+u])
                val_minus = int(image[int(kp_position[ptidx,1])-v, int(kp_position[ptidx,0])+u])
                v_sum = v_sum + (val_plus - val_minus)
                m_10 = m_10 + u * (val_plus + val_minus)
            m_01 = m_01 + v * v_sum
        
        keypoints[ptidx].angle = math.atan2(float(m_01), float(m_10))

def retain_best(keypoints, n_point):
    keypoints.sort(key=lambda kp: kp.response)

    if (n_point >= 0 and len(keypoints) > n_point):
        keypoints = keypoints[:n_point]

    return keypoints


def ofast(image, n_point = 10):

    py_images = []
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    harris_responce = []
    py_kp = []
    py_level = 0

    fast = cv.FastFeatureDetector_create(20, True, cv.FAST_FEATURE_DETECTOR_TYPE_9_16)

    get_pyramid(img, py_images)

    for img_level in py_images:
        kp = fast.detect(img_level, None)
        harris = cv.cornerHarris(img_level, 2, 3, 0.04)

        kp_position = cv.KeyPoint_convert(kp)
        u_max = umax(3)
        ICAngles(img_level, kp, 3, u_max)
        for position in range(len(kp_position)):
            kp[position].response = harris.item((int(kp_position[position,1]), int(kp_position[position,0])))
            kp[position].octave = py_level
            kp[position].pt = (kp_position[position,0] * (py_level+1), kp_position[position,1] * (py_level+1))

        kp = retain_best(kp, n_point)

        py_kp += kp        
        py_level += 1
    
    py_kp = retain_best(py_kp, n_point)

    return py_kp, py_images
    



