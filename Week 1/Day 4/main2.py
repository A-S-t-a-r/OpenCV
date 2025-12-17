import cv2

# 读取图像
img = cv2.imread('Data/Dinosaur.jpg', 0)

# 自适应阈值处理
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 显示结果
cv2.imshow('Adaptive Threshold', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()