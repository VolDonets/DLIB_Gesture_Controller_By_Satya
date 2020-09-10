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

detector_model_one = "models/Hand_Detector_v5_c20.svm"
detector_model_two = "models/Hand_Detector_v6_c20.svm"

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

scale_factor = 2.0
size, center_x = 0, 0
fps = 0
frame_counter = 0
start_time = time.time()

hand_one_detector = dlib.fhog_object_detector(detector_model_one)
hand_two_detector = dlib.fhog_object_detector(detector_model_two)

detectors = [hand_one_detector, hand_two_detector]
names = ['CIRCLE DETECTED', 'HAND DETECTED']


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
    [detections, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, resized_frame, upsample_num_times=1)
    for i in range(len(detections)):
        if confidences[i] >= 0.1:
            x1 = int(detections[i].left() * scale_factor)
            y1 = int(detections[i].top() * scale_factor)
            x2 = int(detections[i].right() * scale_factor)
            y2 = int(detections[i].bottom() * scale_factor)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, '{}: {:.2f}%'.format(names[detector_idxs[i]], confidences[i] * 100), (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
            size = int((x2 - x1) * (y2 - y1))
            center_x = int(x1 + (x2 - x1) / 2)
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
