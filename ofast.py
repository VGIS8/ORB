import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


def pyramid(image, py_image):

    py_image.append(image)
    py_levels = 0

    for level in range(py_levels):
        image = cv.pyrDown(image)
        py_image.append(image)

def insetSort(data):
    for place in range(1,len(data)):
        key = data[place].response
        compared = place - 1
        while compared > -1 and data[compared].response < key:
            data[compared+1].response = data[compared].response
            compared = compared - 1
        data[compared+1].response = key 
    return data

def umax(half_patch_size):
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
    best_keypoint = []
    insetSort(keypoints)
    if (n_point >= 0 and len(keypoints) > n_point):
        if (n_point == 0):
            best_keypoints = []
            return
        for idx in range(n_point):
            best_keypoint.append(keypoints[idx])
        keypoints = []

    keypoints = best_keypoint
    return keypoints



def ofast(image, n_point = 10):

    py_images = []
    img = cv.imread(image)
    harris_responce = []
    py_kp = []
    py_level = 0

    fast = cv.FastFeatureDetector_create(20, True, cv.FAST_FEATURE_DETECTOR_TYPE_9_16)

    pyramid(img, py_images)

    for img_level in py_images:
        kp = fast.detect(img_level, None)
        gray = cv.cvtColor(img_level,cv.COLOR_BGR2GRAY)
        harris = cv.cornerHarris(gray, 2, 3, 0.04)

        kp_position = cv.KeyPoint_convert(kp)
        for position in range(len(kp_position)):

            kp[position].response = harris.item((int(kp_position[position,1]), int(kp_position[position,0])))
            kp[position].octave = py_level

        u_max = umax(3)
        ICAngles(gray, kp, 3, u_max)

        img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        cv.imwrite('kp.png',img2)

        kp = retain_best(kp, n_point)

        img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        cv.imwrite('best_kp.png',img3)

        py_kp.append(kp)
        
        py_level = py_level + 1
    

ofast('elhest.jpg', 1000)