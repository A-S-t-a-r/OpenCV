import cv2 as cv

img=cv.imread('Data/Pikachu.jpg')

#提取ROI
roi=img[200:300,200:300]

#修改ROI
roi[:,:]=[200,0,155]

#放回原图像
img[200:300,200:300]=roi

#缩放
small=cv.resize(img,(300,300))

#翻转
flipped_imag=cv.flip(img,1)

cv.imshow('ROI',roi)
cv.imshow('SMALL',small)
cv.imshow('FLIP',flipped_imag)
cv.waitKey(0)