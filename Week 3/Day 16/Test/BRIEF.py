import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Data/feature_map.png',cv.IMREAD_GRAYSCALE)

# 初始化 FAST 检测器
star = cv.xfeatures2d.StarDetector_create()
 
# 初始化 BRIEF 提取器
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
 
# 使用 STAR 查找关键点
kp = star.detect(img,None)
 
# 使用 BRIEF 计算描述符
kp, des = brief.compute(img, kp)
 
print( brief.descriptorSize() )
print( des.shape )