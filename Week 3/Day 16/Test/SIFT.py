import cv2 as cv
import numpy as np

img = cv.imread('Data/feature_map.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img = cv.drawKeypoints(gray,kp,img)

kp,des = sift.compute(gray,kp)
print(f'Number of keypoints detected: {len(kp)}')

cv.imshow('sift_keypoints',img)
cv.waitKey(0)
cv.destroyAllWindows()