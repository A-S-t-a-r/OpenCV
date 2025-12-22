import cv2 as cv

img = cv.imread('Data/Starsky.jpeg',cv.COLOR_BGR2GRAY)
edges = cv.Canny(img,50,150)
contours,_ = cv.findContours(edges,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv.contourArea(cnt)
    if area < 200:
        continue

    # 多边形逼近
    epsilon = 0.02 * cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)

    # 判断形状
    if len(approx) == 3:
        label = 'Triangle'
    elif len(approx) == 4:
        label = 'Rectangle'
    else:
        label = 'Circle'

    # 绘制边界框和标签
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

cv.imshow('Shapes',img)
cv.waitKey(0)