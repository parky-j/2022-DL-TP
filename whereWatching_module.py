import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd
from sklearn.svm import SVC
from calculator_module import calcFaceAngle
import joblib

# 폰트 색상 지정
blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255) 
# 폰트 지정
font =  cv2.FONT_HERSHEY_PLAIN

#from scipy.spatial import distance as dist
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def create_AtanMean():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    prevTime = time.time()
    totalTime = 0
    currTime = 0

    angle_right_ = []

    # 오른쪽 얼굴 각도 평균값 생성
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("웹캠 인식 불가")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # image로부터 랜드마크 반환
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.putText(image, "Look at the right end of the monitor and press q", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, font_scale=0.5)
                cv2.imshow('hello', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                # 반환된 랜드마크를 image에 덧씌움
                    with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                        prevTime = time.time()
                        totalTime = 0
                        currTime = 0
                        while cap.isOpened():
                            success, image = cap.read()
                            image.flags.writeable = False
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # image로부터 랜드마크 반환
                            results = face_mesh.process(image)

                            image.flags.writeable = True
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            if results.multi_face_landmarks:
                                for face_landmarks in results.multi_face_landmarks:
                                    mypose = face_landmarks
                                    tmpList = []
                                    for idx,lm in enumerate(mypose.landmark):
                                        point=np.array([lm.x, lm.y, lm.z])
                                        tmpList.append(point)

                                angle_ = calcFaceAngle(tmpList)
                                angle_right_.append(angle_)

                                cv2.imshow("Open eyes 5 seconds", image)
                                t = time.time()
                                currTime = t - prevTime
                                prevTime = t

                                totalTime += currTime
                            if totalTime > 5.0:
                                break
                        break                

    cap.release()
    cv2.destroyAllWindows()    

    angle_right = np.mean(angle_right_) 

    prevTime = time.time()
    totalTime = 0
    currTime = 0

    angle_left_ = []
    cap = cv2.VideoCapture(0)
    
    # 왼쪽 얼굴 각도 평균값 생성
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("웹캠 인식 불가")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # image로부터 랜드마크 반환
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.putText(image, "Look at the left end of the monitor and press q", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3)
                cv2.imshow('hello', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                # 반환된 랜드마크를 image에 덧씌움
                    with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                        prevTime = time.time()
                        totalTime = 0
                        currTime = 0
                        while cap.isOpened():
                            success, image = cap.read()
                            image.flags.writeable = False
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # image로부터 랜드마크 반환
                            results = face_mesh.process(image)

                            image.flags.writeable = True
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            if results.multi_face_landmarks:
                                for face_landmarks in results.multi_face_landmarks:
                                    mypose = face_landmarks
                                    tmpList = []
                                    for idx,lm in enumerate(mypose.landmark):
                                        point=np.array([lm.x, lm.y, lm.z])
                                        tmpList.append(point)

                                angle_ = calcFaceAngle(tmpList)
                                angle_left_.append(angle_)

                                cv2.imshow("Open eyes 5 seconds", image)

                                t = time.time()
                                currTime = t - prevTime
                                prevTime = t

                                totalTime += currTime
                            if totalTime > 5.0:
                                break
                        break                

    cap.release()
    cv2.destroyAllWindows()    

    angle_left = np.mean(angle_left_) 
    
    return angle_right, angle_left