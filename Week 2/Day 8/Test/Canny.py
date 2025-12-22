import cv2 as cv

# 读取图像
img = cv.imread('Data/Kapibala.jpeg',cv.IMREAD_GRAYSCALE)

# 应用 Canny 边缘检测
edges = cv.Canny(img,100,200)

# 显示结果
cv.imshow('Canny Edges',edges)
cv.waitKey(0)
cv.destroyAllWindows()