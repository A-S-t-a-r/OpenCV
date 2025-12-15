1. 颜色空间：
   + GRAY:0:纯黑色 ~ 255：纯白色
   + HSV:H:色调;S:饱和度;V:亮度
2. 颜色空间转换：
    a=cv2.cvtColor(src,code)
    src:转换前的初始图像
    code:色彩空间转换码
        ```
        import cv2
        a=cv2.imread("170.jpg")
        b=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)#转换灰度通道
        cv2.imshow("bq",b)
        cv2.waitKey()
        cv2.destroyAllWindows()
        ```
3. 通道拆解：
   + 拆解BGR
     b,g,r=split(bgr_imgar) //bgr_imgar:一副BGR图片
   + 拆解HSV
     h,s,v=split(hsv_imgar) //hsv_imgar:一张hsv图像
4. 通道合并：
   + 合并BGR
     bgr=cv2.merge([b,g,r])
   + 合并HSV
     hsv=cv2.merge[h,s,v]
5. 颜色分割：
        #定义 HSV 中蓝色的范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
 
        #对 HSV 图像进行阈值处理，只获取蓝色
    mask = cv.inRange(hsv, lower_blue, upper_blue)
