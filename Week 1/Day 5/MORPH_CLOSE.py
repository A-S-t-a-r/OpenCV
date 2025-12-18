import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('Data/Elegant_person.jpg',0)

# 定义结构元素
kernel = np.ones((5,5),np.uint8)

# 闭运算
closed_image = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)

# 显示结果
cv.imshow('Closed Image', closed_image)
cv.waitKey(0)
cv.destroyAllWindows()