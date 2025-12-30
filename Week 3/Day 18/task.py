import cv2 as cv
import numpy as np
import glob

# 棋盘格内角点数量
pattern_size = (9, 6)

objp = np.zeros((pattern_size[0] * pattern_size[1],3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints,imgpoints = [],[]
images = glob.glob('Data/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,corners = cv.findChessboardCorners(gray,pattern_size,None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret,mtx,dist, rvecs,tvecs= cv.calibrateCamera(
    objpoints,imgpoints,gray.shape[::-1],None,None)

print('相机内参:\n',mtx)
print('畸变系数:',dist.ravel())