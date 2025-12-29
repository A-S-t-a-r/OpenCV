import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
img1 = cv.imread('Data/feature_map.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Data/map_in_scene.JPG',cv.IMREAD_GRAYSCALE)
 
# 初始化 ORB 检测器
orb = cv.ORB_create()
 
# 使用 ORB 查找关键点和描述符
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# 创建 BFMatcher 对象
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
 
# 匹配描述符。
matches = bf.match(des1,des2)
 
# 按距离顺序对它们进行排序。
matches = sorted(matches, key = lambda x:x.distance)
 
# 绘制前 10 个匹配项。
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()