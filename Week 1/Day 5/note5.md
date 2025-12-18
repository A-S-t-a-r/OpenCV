1. 腐蚀
   ```
    cv2.erode(src, kernel, iterations=1)
   ```
   + src: 输入图像，通常是二值图像
   + kernel: 结构元素，可以自定义或使用 cv2.getStructuringElement() 生成
   + iterations: 腐蚀操作的次数，默认为1
2. 膨胀
   ```
    cv2.dilate(src, kernel, iterations=1)
   ```
3. 开运算
   ```
    cv2.morphologyEx(src, op, kernel)
   ```
   + op: 形态学操作类型，开运算使用 cv2.MORPH_OPEN
4. 闭运算
   + op: cv2.MORPH_CLOSE
5. 形态学梯度
   + op: cv2.MORPH_GRADIENT
6. 结构元素
    ```
    # 定义结构元素
    kernel = np.ones((5,5), np.uint8)
    
    # 矩形内核
    >>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    array([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]], dtype=uint8)

    # 椭圆内核
    >>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    array([[0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0]], dtype=uint8)

    # 十字形内核
    >>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
    array([[0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]], dtype=uint8)

    # 菱形内核
    >>> cv.getStructuringElement(cv.MORPH_DIAMOND,(5,5))
    array([[0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]], dtype=uint8)
    ```
