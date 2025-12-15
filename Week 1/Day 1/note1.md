1. 读取图片：img=cv.imread('...')
2. OpenCV基础模块中HighGUI模块包含：
    图像显示函数: cv.imshow('...',img)、cv.waitKey(0)、cv.destroyAllWindows()
3. 显示图片：cv.imshow('...',img)
4.  RGB和BGR两者相反，通过img[:, :, ::-1]可实现将BGR转为RGB