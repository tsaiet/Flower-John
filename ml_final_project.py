# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:43:37 2021

@author: TSAI ET
"""
import time
import dlib
import cv2
import face_recognition
import numpy as np
import datetime
from PIL import Image, ImageDraw
from math import hypot as hy
import math

#Dlib facial landmarks model的path
predictor_path = "D:/Anaconda3/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"

# 加載已經訓練好的模型路徑，可以是絕對路徑或者相對路徑
weightsPath = "D:/ml/ml/yolov4-tiny_best.weights"
configPath = "D:/ml/ml/yolov4-tiny.cfg"
labelsPath = "D:/ml/ml/yolo_hand.names"
# 初始化一些參數
#strip() 方法用於移除字符串頭尾指定的字符（默認為空格或換行符
#label 為用換行符分割的物體類別
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")  #顏色(2維陣列 labels數*3)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#CUDA加速
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 讀入待檢測的圖像
# 0是代表攝像頭編號，只有一個的話默認為0
capture = cv2.VideoCapture(0)
yolo_num = 1

# 從視訊鏡頭擷取影片
cap = cv2.VideoCapture(0)

#於landmarks上畫圓，標識特徴點
def renderFace(im, landmarks, color=(0, 255, 0), radius=3):
  for p in landmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)

#detector為臉孔偵測，predictor為landmarks偵測
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

t, r, g, b = 0, 180, 0, 0       #t是拉桿

method = 0
temp = -1
shot = 0
def which_method(x):
    global method
    method = cv2.getTrackbarPos('Method ',"Flower John")

def number_to_color(x):
    global t, r, g, b
    t = cv2.getTrackbarPos('Lip ',"Flower John")
    if t == 0:
        r, g, b = 180, 0, 0
    elif t == 1:
        r, g, b = 161, 42, 32
    elif t == 2:
        r, g, b = 223, 109, 111
    elif t == 3:
        r, g, b = 135, 15, 25
    elif t == 4:
        r, g, b = 227, 33, 59
    elif t == 5:
        r, g, b = 206, 30, 29
    elif t == 6:
        r, g, b = 165, 55, 64
    elif t == 7:
        r, g, b = 246, 80, 83
    elif t == 8:
        r, g, b = 252, 94, 59
    elif t == 9:
        r, g, b = 226, 69, 120
    else:
        r, g, b = 0, 0, 0

if __name__ == '__main__': 
    print("按 a 拍照 φ(゜▽゜*)♪")
    print("按 q 離開 ╰（‵□′）╯")
    timmer=0
    flag=0
    while True:
        print("now time:",time.time())
        if time.time()-timmer>=0.8 and flag==1:
            shot=1
            print("shot")
        
        boxes = []
        confidences = []
        classIDs = []
        print("yolo%d" % yolo_num)
        yolo_num = yolo_num+ 1
        # Read the frame
        ref, img = cap.read()
        
        #讀取影像失敗即結束
        if ref == False:
            break
        #2.獲取圖片維度
        (H, W) = img.shape[:2]
        #3.得到 YOLO需要的輸出層
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        rows, cols, _ = img.shape
        
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img)
        
            
        #偵測臉孔
        dets = detector(img, 1)
        
        face_landmarks_list = face_recognition.face_landmarks(img)
        pil_image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
        if temp!=-1:
            method=temp
            temp=-1
        cv2.createTrackbar('Method ',"Flower John", method, 8, which_method)
        
        
        if method == 0:
            for face_landmarks in face_landmarks_list:
        
                d = ImageDraw.Draw(pil_image, 'RGBA')
        
                #讓眉毛變成了一場噩夢
                d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 88))
                d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 88))
                #d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
                #d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
        
                #光澤的嘴脣
                d.polygon(face_landmarks['top_lip'], fill=(r, g, b, 100))
                d.polygon(face_landmarks['bottom_lip'], fill=(r, g, b, 100))
                #d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
                #d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
        
                #閃耀眼睛
                #d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
                #d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30)) 
                
                #塗一些眼線
                d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 50), width=1)
                d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 50), width=1)
        
        
            img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
            
            cv2.createTrackbar('Lip ',"Flower John", t, 10, number_to_color)
        
        elif method == 1:
            #針對相片中的每張臉孔偵測68個landmarks
            for k, d in enumerate(dets):
                shape = predictor(img, d)
                renderFace(img, shape)
        
        elif method == 2:
            pig_nose = cv2.imread("pig_nose.png")
            nose_mask = np.zeros((rows, cols), np.uint8)
            nose_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
                
                # Nose coordinates
                top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                nose_width = int(hy(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 1.7)
                nose_height = int(nose_width * 0.77)
                
                # New nose position
                top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_height / 2))
                bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))
        
                # Adding the new nose
                nose_pig = cv2.resize(pig_nose, (nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)
                img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose
        
        elif method == 3:
            mouth_image = cv2.imread("mask.png")
            mouth_mask = np.zeros((rows, cols), np.uint8)
            mouth_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
        
                # mouth coordinates
                top_mouth = (landmarks.part(28).x, landmarks.part(28).y)
                center_mouth = (landmarks.part(66).x, landmarks.part(66).y)
                left_mouth = (landmarks.part(6).x, landmarks.part(6).y)
                right_mouth = (landmarks.part(10).x, landmarks.part(10).y)
        
                mouth_width = int(hy(left_mouth[0] - right_mouth[0], left_mouth[1] - right_mouth[1]) * 1.9)
                mouth_height = int(mouth_width * 0.9)
        
                # New mouth position
                top_left = (int(center_mouth[0] - mouth_width /2), int(center_mouth[1] - mouth_height / 2))
                bottom_right = (int(center_mouth[0] + mouth_width ),
                               int(center_mouth[1] + mouth_height / 2))
        
        
                # Adding the new mouth
                mouth_pig = cv2.resize(mouth_image, (mouth_width, mouth_height))
                mouth_pig_gray = cv2.cvtColor(mouth_pig, cv2.COLOR_BGR2GRAY)
                _, mouth_mask = cv2.threshold(mouth_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        
                mouth_area = img[top_left[1]: top_left[1] + mouth_height, top_left[0]: top_left[0] + mouth_width]
                mouth_area_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
                final_mouth = cv2.add(mouth_area_no_mouth, mouth_pig)
        
                img[top_left[1]: top_left[1] + mouth_height, top_left[0]: top_left[0] + mouth_width] = final_mouth
        
        elif method == 4:
            forehead_image = cv2.imread("forehead.png")
            forehead_mask = np.zeros((rows, cols), np.uint8)
            forehead_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
        
                # forehead coordinates
                top_forehead = (landmarks.part(27).x, landmarks.part(27).y)
                center_forehead = (landmarks.part(27).x, landmarks.part(27).y)
                left_forehead = (landmarks.part(0).x, landmarks.part(0).y)
                right_forehead = (landmarks.part(16).x, landmarks.part(16).y)
        
                forehead_width = int(hy(left_forehead[0] - right_forehead[0],
                                   left_forehead[1] - right_forehead[1]) )
                forehead_height = int(forehead_width * 0.8)
        
                # New forehead position
                top_left = (int(center_forehead[0] - forehead_width /2), int(center_forehead[1] - 9*forehead_height/10))
                bottom_right = (int(center_forehead[0] + forehead_width ), int(center_forehead[1] + forehead_height / 2))
        
                # Adding the new forehead
                forehead_pig = cv2.resize(forehead_image, (forehead_width, forehead_height))
                forehead_pig_gray = cv2.cvtColor(forehead_pig, cv2.COLOR_BGR2GRAY)
                _, forehead_mask = cv2.threshold(forehead_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        
                forehead_area = img[top_left[1]: top_left[1] + forehead_height, top_left[0]: top_left[0] + forehead_width]
                forehead_area_no_forehead = cv2.bitwise_and(forehead_area, forehead_area, mask=forehead_mask)
                final_forehead = cv2.add(forehead_area_no_forehead, forehead_pig)
        
                img[top_left[1]: top_left[1] + forehead_height, top_left[0]: top_left[0] + forehead_width] = final_forehead
        
        elif method == 5:
            princess = cv2.imread("princess.png")
            princess_mask = np.zeros((rows, cols), np.uint8)
            princess_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
                
                # Nose coordinates
                top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                nose_width = int(hy(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) )
                nose_height = int(nose_width )
                
                # New nose position
                top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_height * 3))
                bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))
        
                # Adding the new nose
                nose_pig = cv2.resize(princess, (nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, princess_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=princess_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)
                img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose
        
        elif (method == 6):
            heart = cv2.imread("main.png")
            heart_mask = np.zeros((rows, cols), np.uint8)
            heart_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
                
                # Nose coordinates
                top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                nose_width = int(hy(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) )
                nose_height = int(nose_width )
                
                # New nose position
                top_left = (int(center_nose[0] - nose_width *2), int(center_nose[1] - nose_height /2))
                bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))
                top_right = (int(center_nose[0] + nose_width *1.5), int(center_nose[1] - nose_height /2))
                
                # Adding the new nose
                nose_pig = cv2.resize(heart, (nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, heart_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                
                nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=heart_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)
                img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose
                
                nose_area1 = img[top_right[1]: top_right[1] + nose_height, top_right[0]: top_right[0] + nose_width]
                nose_area_no_nose1 = cv2.bitwise_and(nose_area1, nose_area1, mask=heart_mask)
                final_nose1 = cv2.add(nose_area_no_nose1, nose_pig)
                img[top_right[1]: top_right[1] + nose_height, top_right[0]: top_right[0] + nose_width] = final_nose1
        
        elif method == 7:
            left_ear = cv2.imread("left_ear.png")
            right_ear = cv2.imread("right_ear.png")
            ear_mask = np.zeros((rows, cols), np.uint8)
            ear_mask.fill(0)
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(img)
            for face in faces:
                landmarks = predictor(gray_frame, face)
                
                # Nose coordinates
                top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                nose_width = int(hy(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]))
                nose_height = int(nose_width )
                
                # New nose position
                top_left = (int(center_nose[0] - nose_width * 2), int(center_nose[1] - nose_height * 4))
                bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))
                top_right = (int(center_nose[0] + nose_width * 2), int(center_nose[1] - nose_height * 4))
                # Adding the new nose
                nose_pig = cv2.resize(left_ear, (nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, ear_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=ear_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)
                img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose
                
                nose_pig1 = cv2.resize(right_ear, (nose_width, nose_height))
                nose_pig_gray1 = cv2.cvtColor(nose_pig1, cv2.COLOR_BGR2GRAY)
#                _, ear_mask = cv2.threshold(nose_pig_gray1, 25, 255, cv2.THRESH_BINARY_INV)
                nose_area1 = img[top_right[1]: top_right[1] + nose_height, top_right[0]: top_right[0] + nose_width]
                nose_area_no_nose1 = cv2.bitwise_and(nose_area1, nose_area1, mask=ear_mask)
                final_nose1 = cv2.add(nose_area_no_nose1, nose_pig1)
                img[top_right[1]: top_right[1] + nose_height, top_right[0]: top_right[0] + nose_width] = final_nose1
        
        else:
            flower_image = cv2.imread("flower.png")
            flower_mask = np.zeros((rows, cols), np.uint8)
            flower_mask.fill(0)
            for face in faces:
                landmarks = predictor(gray_frame, face)
        
                # flower coordinates
                top_flower = (landmarks.part(27).x, landmarks.part(27).y)
                center_flower = (landmarks.part(33).x, landmarks.part(33).y)
                left_flower = (landmarks.part(49).x, landmarks.part(49).y)
                right_flower = (landmarks.part(53).x, landmarks.part(53).y)
        
                flower_width = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1]) *0.9))
                flower_height = int(flower_width*0.9 )
        
                # New flower position
                top_left = (int(center_flower[0] + 2.5*flower_width ),int(center_flower[1] - flower_height*0.5))
                bottom_right = (int(center_flower[0] + flower_width ),
                               int(center_flower[1] + flower_height ))
        
        
                # Adding the new flower
                flower_pig = cv2.resize(flower_image, (flower_width, flower_height))
                flower_pig_gray = cv2.cvtColor(flower_pig, cv2.COLOR_BGR2GRAY)
                _, flower_mask = cv2.threshold(flower_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area = img[top_left[1]: top_left[1] + flower_height, top_left[0]: top_left[0] + flower_width]
                flower_area_no_flower = cv2.bitwise_and(flower_area, flower_area, mask=flower_mask)
                final_flower = cv2.add(flower_area_no_flower, flower_pig)
        
                img[top_left[1]: top_left[1] + flower_height, top_left[0]: top_left[0] + flower_width] = final_flower
        
        ######################################################################################
        
                flower_width2 = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1]) *0.8))
                flower_height2 = int(flower_width2 *0.8)
        
                # New flower position
                top_left2 = (int(center_flower[0] - 4*flower_width2 ),int(center_flower[1] - 3*flower_height2*0.5))
                bottom_right2 = (int(center_flower[0] + flower_width2 ), int(center_flower[1] + flower_height2 ))
        
        
                # Adding the new flower
                flower_pig2 = cv2.resize(flower_image, (flower_width2, flower_height2))
                flower_pig_gray2 = cv2.cvtColor(flower_pig2, cv2.COLOR_BGR2GRAY)
                _, flower_mask2 = cv2.threshold(flower_pig_gray2, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area2 = img[top_left2[1]: top_left2[1] + flower_height2, top_left2[0]: top_left2[0] + flower_width2]
                flower_area_no_flower2 = cv2.bitwise_and(flower_area2, flower_area2, mask=flower_mask2)
                final_flower2 = cv2.add(flower_area_no_flower2, flower_pig2)
        
                img[top_left2[1]: top_left2[1] + flower_height2, top_left2[0]: top_left2[0] + flower_width2] = final_flower2
                ##################################################################3
                flower_width3 = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1])*0.7 ))
                flower_height3 = int(flower_width3*0.7 )
        
                # New flower position
                top_left3 = (int(center_flower[0] - flower_width3 ),int(center_flower[1] - 11*flower_height3))
                bottom_right3 = (int(center_flower[0] + flower_width3 ),
                               int(center_flower[1] + flower_height3 ))
        
        
                # Adding the new flower
                flower_pig3 = cv2.resize(flower_image, (flower_width3, flower_height3))
                flower_pig_gray3 = cv2.cvtColor(flower_pig3, cv2.COLOR_BGR2GRAY)
                _, flower_mask3 = cv2.threshold(flower_pig_gray3, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area3 = img[top_left3[1]: top_left3[1] + flower_height3,top_left3[0]: top_left3[0] + flower_width3]
                flower_area_no_flower3 = cv2.bitwise_and(flower_area3, flower_area3, mask=flower_mask3)
                final_flower3 = cv2.add(flower_area_no_flower3, flower_pig3)
        
                img[top_left3[1]: top_left3[1] + flower_height3, top_left3[0]: top_left3[0] + flower_width3] = final_flower3
                ##################################################################3
                flower_width4 = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1]) *0.7))
                flower_height4 = int(flower_width4*0.7 )
        
                # New flower position
                top_left4 = (int(center_flower[0] + flower_width4 ),int(center_flower[1] - 10*flower_height4))
                bottom_right4 = (int(center_flower[0] + flower_width4 ),
                               int(center_flower[1] + flower_height4 ))
        
        
                # Adding the new flower
                flower_pig4 = cv2.resize(flower_image, (flower_width4, flower_height4))
                flower_pig_gray4 = cv2.cvtColor(flower_pig4, cv2.COLOR_BGR2GRAY)
                _, flower_mask4 = cv2.threshold(flower_pig_gray4, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area4 = img[top_left4[1]: top_left4[1] + flower_height4, top_left4[0]: top_left4[0] + flower_width4]
                flower_area_no_flower4 = cv2.bitwise_and(flower_area4, flower_area4, mask=flower_mask4)
                final_flower4 = cv2.add(flower_area_no_flower4, flower_pig4)
        
                img[top_left4[1]: top_left4[1] + flower_height4, top_left4[0]: top_left4[0] + flower_width4] = final_flower4
                ##################################################################3
                flower_width5 = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1])*0.7 ))
                flower_height5 = int(flower_width5*0.7 )
        
                # New flower position
                top_left5 = (int(center_flower[0] +3* flower_width5 ),int(center_flower[1] - 6*flower_height5))
                bottom_right5 = (int(center_flower[0] + flower_width5 ), int(center_flower[1] + flower_height5 ))
        
        
                # Adding the new flower
                flower_pig5 = cv2.resize(flower_image, (flower_width5, flower_height5))
                flower_pig_gray5 = cv2.cvtColor(flower_pig5, cv2.COLOR_BGR2GRAY)
                _, flower_mask5 = cv2.threshold(flower_pig_gray5, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area5 = img[top_left5[1]: top_left5[1] + flower_height5, top_left5[0]: top_left5[0] + flower_width5]
                flower_area_no_flower5 = cv2.bitwise_and(flower_area5, flower_area5, mask=flower_mask5)
                final_flower5 = cv2.add(flower_area_no_flower5, flower_pig5)
        
                img[top_left5[1]: top_left5[1] + flower_height5, top_left5[0]: top_left5[0] + flower_width5] = final_flower5
                 ##################################################################3
                flower_width6 = int((hy(left_flower[0] - right_flower[0], left_flower[1] - right_flower[1])*0.7 ))
                flower_height6 = int(flower_width6 )
        
                # New flower position
                top_left6 = (int(center_flower[0] -4* flower_width6 ),int(center_flower[1] - 4*flower_height6))
                bottom_right6 = (int(center_flower[0] + flower_width6 ),
                               int(center_flower[1] + flower_height6 ))
        
        
                # Adding the new flower
                flower_pig6 = cv2.resize(flower_image, (flower_width6, flower_height6))
                flower_pig_gray6 = cv2.cvtColor(flower_pig6, cv2.COLOR_BGR2GRAY)
                _, flower_mask6 = cv2.threshold(flower_pig_gray6, 26, 266, cv2.THRESH_BINARY_INV)
        
                flower_area6 = img[top_left6[1]: top_left6[1] + flower_height6, top_left6[0]: top_left6[0] + flower_width6]
                flower_area_no_flower6 = cv2.bitwise_and(flower_area6, flower_area6, mask=flower_mask6)
                final_flower6 = cv2.add(flower_area_no_flower6, flower_pig6)
        
                img[top_left6[1]: top_left6[1] + flower_height6,
                            top_left6[0]: top_left6[0] + flower_width6] = final_flower6
                ######################################################################################
        
                flower_width7 = int((hy(left_flower[0] - right_flower[0],
                                   left_flower[1] - right_flower[1]) *0.8))
                flower_height7 = int(flower_width2 *0.8)
        
                # New flower position
                top_left7 = (int(center_flower[0] - 3*flower_width7 ),int(center_flower[1] - 15*flower_height7*0.5))
                bottom_right7 = (int(center_flower[0] + flower_width7 ),
                               int(center_flower[1] + flower_height7 ))
        
        
                # Adding the new flower
                flower_pig7 = cv2.resize(flower_image, (flower_width7, flower_height7))
                flower_pig_gray7 = cv2.cvtColor(flower_pig7, cv2.COLOR_BGR2GRAY)
                _, flower_mask7 = cv2.threshold(flower_pig_gray7, 25, 255, cv2.THRESH_BINARY_INV)
        
                flower_area7 = img[top_left7[1]: top_left7[1] + flower_height7, top_left7[0]: top_left7[0] + flower_width7]
                flower_area_no_flower7 = cv2.bitwise_and(flower_area7, flower_area7, mask=flower_mask7)
                final_flower7 = cv2.add(flower_area_no_flower7, flower_pig7)
        
                img[top_left7[1]: top_left7[1] + flower_height7, top_left7[0]: top_left7[0] + flower_width7] = final_flower7
        
        imgg = img
        
        blob = cv2.dnn.blobFromImage(imgg, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        # 疊代每個輸出層，總共三個
        for output in layerOutputs:
            # 對每個檢測進行循環
            for detection in output:
                # 提取類別ID和置信度
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                #過濾掉那些置信度較小的檢測結果
                if confidence > 0.5:
                    # 將邊界框的坐標還原至與原圖片相匹配，記住YOLO返回的是 
                    # 邊界框的中心坐標以及邊界框的寬度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 計算邊界框的左上角位置
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新邊界框，置信度（機率）以及類別
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # 使用非極大值抑制方法抑制弱、重疊邊界框
        '''
        confidence_thre：0-1，置信度（機率/打分）閾值，即保留機率大於這個值的邊界框，默認為0.5 
        nms_thre：非極大值抑制的閾值，默認為0.3
        '''
        
        
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        # 確保至少一個邊界框
        if len(idxs) > 0:
            print("in class")
            # 疊代每個邊界框
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 繪製邊界框以及在左上角添加類別標籤和置信度
                color = [int(c) for c in COLORS[classIDs[i]]]
                
                if LABELS[classIDs[i]] == "OK":#只檢測eat類別
                    timmer=time.time()
                    flag=1
                    print(" Label: %s:(x,y,w,h)=(%d,%d,%d,%d)" % (LABELS[classIDs[i]], x, y, w, h))
                    print(timmer)
                elif LABELS[classIDs[i]] == "heart" :#只檢測eat類別
                    temp = 6
                    print(" Label: %s:(x,y,w,h)=(%d,%d,%d,%d)" % (LABELS[classIDs[i]], x, y, w, h))
                    cv2.rectangle(imgg, (x, y), (x + w, y + h), color, 2)##左上，右下
                    text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i]) 
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) 
                    cv2.rectangle(imgg, (x, y-text_h-baseline), (x + text_w, y), color, -1) 
                    cv2.putText(imgg, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                elif LABELS[classIDs[i]] == "cat" :#只檢測eat類別
                    temp = 7
                    print(" Label: %s:(x,y,w,h)=(%d,%d,%d,%d)" % (LABELS[classIDs[i]], x, y, w, h))
                    cv2.rectangle(imgg, (x, y), (x + w, y + h), color, 2)##左上，右下
                    text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i]) 
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) 
                    cv2.rectangle(imgg, (x, y-text_h-baseline), (x + text_w, y), color, -1) 
                    cv2.putText(imgg, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 顯示成果
        cv2.namedWindow("Flower John", cv2.WINDOW_NORMAL)  #正常視窗大小
        cv2.imshow("Flower John", imgg)                      #秀出圖片
            
        key = cv2.waitKey(200)
        
        if (key in [ord('a'), 1048673]) | (shot == 1):
            ts = datetime.datetime.now()
            filename = "D:/ml/{}.png".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            cv2.imwrite(filename, img)
            print('拍了一張照片❤')
            cv2.putText(img, "save image!", (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Flower John", imgg)
            cv2.waitKey(3000)
            flag=0
            shot = 0
            
        elif key in [ord('q'), 1048673]:
            cv2.destroyAllWindows()
            break
     
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()