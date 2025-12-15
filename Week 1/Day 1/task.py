import cv2 as cv

# 读取图片
imag = cv.imread('data/bird.jpg')
if imag is None:
    print("无法读取图片")
    exit()

#显示图片
cv.imshow('My Image',imag)
cv.waitKey(0)
cv.destroyAllWindows()

#保存图片
cv.imwrite('data/bird_copy.png',imag)