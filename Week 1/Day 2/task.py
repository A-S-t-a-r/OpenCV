import cv2 as cv

img = cv.imread('Data/Green Ball.png')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

#绿色范围
low =(35,60,40)
high = (99,120,200)

mask = cv.inRange(img,low,high)
result = cv.bitwise_and(img, img, mask=mask)

cv.imshow('Original', img)
cv.imshow('Mask', mask)

cv.imshow('Result', result)
cv.waitKey(0)