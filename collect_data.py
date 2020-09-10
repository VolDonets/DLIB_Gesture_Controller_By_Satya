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

# automate the annotation process

cleanup = True

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cv2.resizeWindow('frame', 1920, 1080)
cv2.moveWindow("frame", 0, 0)

cap = cv2.VideoCapture(0)

x1, y1 = 0, 0

window_width = 190  # 140
window_height = 190

skip_frames = 3
frame_gap = 0

directory = 'train_images_h'
box_file = 'boxes_h.txt'

if cleanup:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    open(box_file, 'w').close()
    counter = 0
elif os.path.exists(box_file):
    with open(box_file, 'r') as text_file:
        box_content = text_file.read()
    counter = int(box_content.split(':')[-2].split(',')[-1])
fr = open(box_file, 'a')

if not os.path.exists(directory):
    os.mkdir(directory)

initial_wait = 0

while (True):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    if initial_wait > 60:
        frame_gap += 1
        if x1 + window_width < frame.shape[1]:
            x1 += 4
            time.sleep(0.1)
        elif y1 + window_height + 270 < frame.shape[1]:
            y1 += 80
            x1 = 0
            frame_gap = 0
            initial_wait = 0
        else:
            break

    else:
        initial_wait += 1
    if frame_gap == skip_frames:
        img_name = str(counter) + '.png'
        img_full_name = directory + '/' + str(counter) + '.png'
        cv2.imwrite(img_full_name, orig)
        fr.write('{}:({},{},{},{}),'.format(counter, x1, y1, x1 + window_width, y1 + window_height))
        counter += 1
        frame_gap = 0
    cv2.rectangle(frame, (x1, y1), (x1 + window_width, y1 + window_height), (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
fr.close()
