import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('Data/Elegant_person.jpg',0)

# 定义结构元素
kernel = np.ones((5,5),np.uint8)

# 腐蚀操作
eroded_image = cv.erode(img,kernel,iterations=1)

# 显示结果
cv.imshow('Eroded Image',eroded_image)
cv.waitKey(0)
cv.destroyAllWindows()