import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from calculator_module import euclideanDistance, eye_aspect_ratio
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

# right eyes indices
RIGHT_EYE=[33, 160, 158, 133, 153, 144] 

# Left eyes indices 
LEFT_EYE =[263, 387, 385, 362, 380, 373] 

# 감은 눈, 뜬 눈에서 EAR 계산해 각각을 배열로써 return
def create_EARlist():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    prevTime = time.time()
    totalTime = 0
    currTime = 0

    open_ear = []
    close_ear = []

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

                cv2.putText(image, "enter q and open eyes 5 sec", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3)
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

                                ear = eye_aspect_ratio(image, tmpList, RIGHT_EYE, LEFT_EYE)
                                open_ear.append(ear)

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
    
    cap = cv2.VideoCapture(0)

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

                cv2.putText(image, "enter q and close eyes 5 sec", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3)
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

                                ear = eye_aspect_ratio(image, tmpList, RIGHT_EYE, LEFT_EYE)
                                close_ear.append(ear)

                                cv2.imshow("enter q and close eyes 5 sec", image)

                                t = time.time()
                                currTime = t - prevTime
                                prevTime = t

                                totalTime += currTime
                            if totalTime > 5.0:
                                break
                        break                

    cap.release()
    cv2.destroyAllWindows()
    
    return close_ear, open_ear
    
# 로지스틱 모델 생성 / 저장
def create_logistic_model():
    close_ear, open_ear = create_EARlist()
    
    classifier = LogisticRegression()

    ones = np.ones_like(close_ear)
    zeros = np.zeros_like(open_ear)

    x_ = np.concatenate((close_ear, open_ear))
    y_ = np.concatenate((ones, zeros))

    X = x_.reshape(-1,1)
    y = y_.reshape(-1,1)

    classifier.fit(X,y)

    joblib.dump(classifier, './pkl/ear_logistic.pkl')