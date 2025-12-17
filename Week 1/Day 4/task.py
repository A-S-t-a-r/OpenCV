import cv2 as cv

img = cv.imread('Data/Dinosaur.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 全局阈值
_,thresh1 = cv.threshold(gray,127,25,cv.THRESH_BINARY)

# OTSU阈值
_,thresh2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# 自适应阈值
thresh3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
cv.THRESH_BINARY,11,2)

cv.imshow('Global',thresh1)
cv.imshow('OTSU',thresh2)
cv.imshow('Adaptive',thresh3)
cv.waitKey(0)