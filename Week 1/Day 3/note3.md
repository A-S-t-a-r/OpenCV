1. ROI
   +  提取ROI
    ```
    roi = img[0:100, 0:100]
    ```
   + 修改ROI
    ```
    #将 ROI 区域设置为绿色
    roi[:, :] = [0, 255, 0] 
    #将修改后的 ROI 放回原图像
    img[0:100, 0:100] = roi
    ```
2. 缩放、平移、旋转、翻转
    ```
    # 缩放
    resized_img = cv.resize(img, (200, 200)) 
    # 旋转
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转 45 度
    rotated_img = cv2.warpAffine(img, M, (w, h))
    # 平移
    M = np.float32([[1, 0, 100], [0, 1, 50]])  # 右100像素，下50像素
    translated_img = cv2.warpAffine(img, M, (w, h))
    # 翻转
    flipped_img = cv2.flip(img, 1)  # 水平翻转
    ```