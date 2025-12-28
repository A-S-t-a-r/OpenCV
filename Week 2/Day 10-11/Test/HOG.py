import cv2 as cv
import numpy as np
from skimage import feature,exposure

img = cv.imread('Data/Athlete.jpg')

# γ矫正(归一化)
img_float = img.astype(np.float32)/255.0
jiaozheng = np.power(img_float,1.5)

# 转灰度图
jiaozheng_unit8 = (jiaozheng*255).astype(np.uint8)
gray = cv.cvtColor(jiaozheng_unit8,cv.COLOR_BGR2GRAY)

# 计算x和y方向的梯度
gx = cv.Sobel(gray,cv.CV_32F,1,0,ksize=1)
gy = cv.Sobel(gray,cv.CV_32F,0,1,ksize=1)

# 计算合梯度的幅值和方向（角度）
mag,angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

# HOG 
fd, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

cv.imshow('jiaozheng', jiaozheng)
cv.imshow('gray', gray)
cv.imshow('tidufudu',angle)
cv.imshow('HOG',hog_image_rescaled)
cv.waitKey(0)==ord('q')
cv.destroyAllWindows()
