1. Canny 边缘检测
   ```
   edges = cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
   ```
   + threshold1：低阈值
   + threshold2：高阈值
   + apertureSize：Sobel 算子的孔径大小，默认为 3
   + L2gradient：是否使用 L2 范数计算梯度幅值，默认为 False
2. Sobel 算子
   ```
   dst = cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
   ```
   + ddepth：输出图像的深度，通常 cv2.CV_64F
   + dx：x 方向导数阶数
   + dy：y 方向导数阶数
   + ksize：Sobel 核的大小，默认为 3
   + scale：缩放因子，默认为 1
   + delta：可选的 delta 值，默认为 0
   + borderType：边界填充类型，默认为 cv2.BORDER_DEFAULT
3. Laplacian 算子
   ```
   dst = cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
   ```
   + ksize：Laplacian 核的大小，默认为 1