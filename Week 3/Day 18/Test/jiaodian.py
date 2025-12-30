import numpy as np
import cv2 as cv
import glob
 
# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# 准备对象点，例如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# 用于存储所有图像中的对象点和图像点的数组。
objpoints = [] # 3d 点在真实世界空间中
imgpoints = [] # 2d 点在图像平面中。
 
images = glob.glob('Data/left/*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
# 查找棋盘角点
ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 
# 如果找到，添加对象点，图像点（在细化它们之后）
if ret == True:
    objpoints.append(objp)
 
corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners2)
 
# 绘制并显示角点
cv.drawChessboardCorners(img, (7,6), corners2, ret)

cv.imshow('img', img)

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 校正图像
img = cv.imread('Data/left/left12.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 畸变校正
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# 裁剪图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult.png', dst)

cv.waitKey(0)
 
cv.destroyAllWindows()