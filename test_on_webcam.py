import dlib
import glob
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil

file_name = 'models/Hand_Detector_v7_c10.svm'
detector = dlib.simple_object_detector(file_name)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

scale_factor = 2.0
size, center_x = 0, 0
fps = 0
frame_counter = 0
start_time = time.time()

while (True):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    copy = frame.copy()
    new_width = int(frame.shape[1] / scale_factor)
    new_height = int(frame.shape[0] / scale_factor)
    resized_frame = cv2.resize(copy, (new_width, new_height))
    detections = detector(resized_frame)
    for detection in (detections):
        x1 = int(detection.left() * scale_factor)
        y1 = int(detection.top() * scale_factor)
        x2 = int(detection.right() * scale_factor)
        y2 = int(detection.bottom() * scale_factor)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Hand Detected', (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        size = int((x2 - x1) * (y2 - y1))
        center_x = x2 - x1 // 2
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
