import numpy as np
import cv2 as cv

img = cv.imread('Data/feature_map.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray,2,3,0.04)

# 标记角点
dst = cv.dilate(dst,None)

# 最佳阈值
img[dst>0.01*dst.max()]=[0,0,255]
 
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()