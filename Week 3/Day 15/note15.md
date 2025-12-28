1. Harris角点检测:
   ```
   cv.cornerHarris(	img, blockSize, ksize, k )
   ```
   + img: 输入图像(灰度+float32) 
   + blockSize: 角点检测的邻域大小
   + ksize: Sobel 导数的孔径
   + k: 自由参数
2. Shi‐Tomasi角点检测:
   ```
   cv.goodFeaturesToTrack( img,corners,qualityLevel,minDistance)
   ```
   + corners：角点的输出数量
   + qualityLevel：最小质量参数
   + minDistance：最小几何距离
3. FAST角点检测:
   ```
   cv.FastFeatureDetector_create()
   ```
4. 通用角点检测:...
5. 亚像素角点检测:
   ```
   cv.cornerSubPix(img,corners,winSize,zeroZone,criteria)
   ```
   + winSize: 搜索窗口变长一半
   + zeroZone： 搜索区域中间死区大小一半
   + criteria： 角点细化迭代过程的终止准则