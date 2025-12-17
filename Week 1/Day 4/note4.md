1. 简单阈值
    ```
    retval, dst = cv2.threshold(src, thresh, maxval, type)
    ```
    + src: 图像
    + thresh: 设定阈值
    + maxval: 超过阈值，赋新值
    + type: 处理类型
2. 自适应阈值
    ```
    dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    ```
    1. adaptiveMethod常见类型：
    + cv2.ADAPTIVE_THRESH_MEAN_C: 邻域均值减 C
    + cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 邻域加权均值减C，权重由高斯函数确定
    2. thresholdType: 处理类型为 cv2.THRESH_BINARY 或 cv2.THRESH_BINARY_INV
    3. blockSize: 计算阈值时使用的邻域大小，必须为奇数
    4. C: 常数
3. OTSU自动阈值
   ```
   retval, dst = cv2.threshold(src, thresh, maxval, type)
   ```
   + type: cv2.THRESH_BINARY 或 cv2.THRESH_BINARY_INV，并加上 cv2.THRESH_OTSU。
4. 阈值类型
   1.  简单常见类型：
    + cv2.THRESH_BINARY: 像大于阈，则赋 maxval，否赋 0
    + cv2.THRESH_BINARY_INV: 与上者相反，像大于值，则赋0，否赋maxval
    + cv2.THRESH_TRUNC: 像大于阈，则赋阈，否则不变
    + cv2.THRESH_TOZERO: 像大于阈，则不变，否则赋 0
    + cv2.THRESH_TOZERO_INV: 与上者相反，像大于阈，则赋 0，否不变
   2. 自适应常见类型：见上
   3. OTSU常见类型：同上