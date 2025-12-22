1. 轮廓查找：findContours()
   ```
   contours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
   ```
   + contours: 检测到的轮廓列表
   + hierarchy: 轮廓的层次结构信息

   + mode: 检索模式：
     + cv2.RETR_EXTERNAL: 只最外层
     + cv2.RETR_LIST: 检测所有，但不建立层次关系
     + cv2.RETR_TREE: 检测所有，并建立完整层次结构
   + method: 轮廓近似：
     + cv2.CHAIN_APPROX_NONE: 存储所有的轮廓点
     + cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，只保留端点
   + contours: 输出的轮廓列表，每个轮廓是一个点集
   + offset: 可选参数，轮廓点的偏移量
  
2. 轮廓绘制：drawContours()
   ```
   cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
   ```
   + contourIdx: 要绘制的轮廓索引，若为负，则绘制所有
   + color: 轮廓颜色
   + thickness: 轮廓线厚度，若为负数，则填充内部
   + lineType: 线型
   + maxLevel: 绘制最大层次深度
   
3. 轮廓属性：
   1. 面积
   ```
   area = cv2.contourArea(contour[, oriented])
   ```
   2. 周长
   ```
   length = cv2.arcLength(curve, closed)
   ```
   + closed: 布尔值，表示轮廓是否闭合
   3. 边界框
   ```
   x, y, w, h = cv2.boundingRect(points)
   ```
   4. 最小外接矩形
   ```
   rect = cv2.minAreaRect(points)
   ```
   5. 最小外接圆
   ```
   (center, radius) = cv2.minEnclosingCircle(points)
   ```
4. 轮廓近似：approxPolyDP()
   ```
   approx = cv2.approxPolyDP(curve, epsilon, closed)
   ```
   + epsilon: 近似精度，值越小，近似越精确
5. 凸包：convexHull()
6. 霍夫变换：直线检测与圆检测