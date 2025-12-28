import cv2 as cv

src = cv.imread("Data/People.jpg")

# hog特征描述
hog = cv.HOGDescriptor()
# 创建SVM检测器
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# 检测行人
(rects, weights) = hog.detectMultiScale(src,winStride=(4, 4),padding=(8, 8),scale=1.25,useMeanshiftGrouping=False)

for (x, y, w, h) in rects:
    cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("hog-people", src)
cv.waitKey(0)
cv.destroyAllWindows()