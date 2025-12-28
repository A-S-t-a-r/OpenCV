from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
 
source_window = 'Image'
maxTrackbar = 25
rng.seed(12345)
 
def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)

     # Shi-Tomasi 算法的参数
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
 
    # 复制源图像
    copy = np.copy(src)
 
    # 应用角点检测
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
    blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
 
    # 绘制检测到的角点
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
 
    # 展示结果
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)
 
    # Set the needed parameters to find the refined corners
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
 
    # Calculate the refined corner locations
    corners = cv.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)
 
    # Write them down
    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "] (", corners[i,0,0], ",", corners[i,0,1], ")")
 
# 加载源图像并将其转换为灰度图像 
src = cv.imread("Data/feature_map.png")
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
 
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
 
# 创建窗口和轨迹栏
cv.namedWindow(source_window)
maxCorners = 10 # initial threshold
cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
cv.imshow(source_window, src)
goodFeaturesToTrack_Demo(maxCorners)
 
cv.waitKey()