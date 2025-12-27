import cv2 as cv
import numpy as np
import time
import os

cap = cv.VideoCapture('Data/test.mp4')
chars = [' ','#']

os.system('cls' if os.name == 'nt' else 'clear')

while True:
    ret,frame = cap.read()
    if not ret or frame is None:
        break
    if frame.ndim == 3:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    else:
        gray = frame
    small = cv.resize(gray,(80,30))

    output = ''
    for row in small:
        for pixel in row:
            output += '#' if pixel > 127 else ' '
        output += '\n'
        
    print('\033[H' + output, end='')

    time.sleep(0.03)

cap.release()