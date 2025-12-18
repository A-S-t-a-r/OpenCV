import cv2 as cv

img = cv.imread('Data/Wood.jpg',0)

blur = cv.blur(img,(5,5))
gblur = cv.GaussianBlur(img,(5,5),0)
mblur = cv.medianBlur(img,5)
bblur = cv.bilateralFilter(img,9,75,75)

cv.imshow('Original', img)
cv.imshow('Mean', blur)
cv.imshow('Gaussian', gblur)
cv.imshow('Median', mblur)
cv.imshow('Bilateral', bblur)
cv.waitKey(0)