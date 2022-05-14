#!/usr/bin/env python
# coding: utf-8
import cv2
import mediapipe as mp
import math
import numpy as np

def calcFaceAngle(landmarks):
    det = math.atan2(landmarks[234][0]-landmarks[454][0], landmarks[234][2]-landmarks[454][2])
    det = det*180/math.pi
    return np.abs(det)

def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# return EAR
def eye_aspect_ratio(image, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]][:2]
    rh_left = landmarks[right_indices[3]][:2]
    rv_top1 = landmarks[right_indices[2]][:2]
    rv_top2 = landmarks[right_indices[1]][:2]
    rv_bottom1 = landmarks[right_indices[4]][:2]
    rv_bottom2 = landmarks[right_indices[5]][:2]
    
    lh_right = landmarks[left_indices[0]][:2]
    lh_left = landmarks[left_indices[3]][:2]
    lv_top1 = landmarks[left_indices[2]][:2]
    lv_top2 = landmarks[left_indices[1]][:2]
    lv_bottom1 = landmarks[left_indices[4]][:2]
    lv_bottom2 = landmarks[left_indices[5]][:2]
    
    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance1 = euclideanDistance(rv_top1, rv_bottom1)
    rvDistance2 = euclideanDistance(rv_top2, rv_bottom2)
    rvDistance = rvDistance1 + rvDistance2
    
    lvDistance1 = euclideanDistance(lv_top1, lv_bottom1)
    lvDistance2 = euclideanDistance(lv_top2, lv_bottom2)
    lhDistance = euclideanDistance(lh_right, lh_left)
    lvDistance = lvDistance1 + lvDistance2

    reRatio = rvDistance/rhDistance/2
    leRatio = lvDistance/lhDistance/2

    ear = (reRatio+leRatio)/2
    
    return reRatio