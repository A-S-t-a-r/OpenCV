import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
img1 = cv.imread('Data/feature_map.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Data/map_in_scene.JPG',cv.IMREAD_GRAYSCALE)
 
# 初始化 SIFT 检测器
sift = cv.SIFT_create()
 
# 使用 SIFT 查找关键点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
# 带有默认参数的 BFMatcher
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
# 应用比率测试
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn 期望列表作为匹配项的列表。
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()