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
 
# FLANN 参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # 或传递空字典
 
flann = cv.FlannBasedMatcher(index_params,search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# 需要仅绘制良好的匹配项，因此请创建一个掩码
matchesMask = [[0,0] for i in range(len(matches))]
 
# 根据 Lowe 的论文进行比率测试
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
 
draw_params = dict(matchColor = (0,255,0),
singlePointColor = (255,0,0),
matchesMask = matchesMask,
flags = cv.DrawMatchesFlags_DEFAULT)
 
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
 
plt.imshow(img3,),plt.show()