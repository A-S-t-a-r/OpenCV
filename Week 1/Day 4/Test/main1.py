import cv2

# 读取图像
img = cv2.imread('Data/Dinosaur.jpg', 0)

# 简单阈值处理
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 显示结果
cv2.imshow('Binary Threshold', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()