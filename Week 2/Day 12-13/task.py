import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #蓝色范围（可调整）
    mask = cv.inRange(hsv,(100,80,40),(130,255,255))

    #连通域分析
    num,labels, stats,centroids= cv.connectedComponentsWithStats(mask)

    for i in range(1,num): #跳过背景
        x,y,w,h,area = stats[i]
        cx,cy = centroids[i]
        if area > 500: #过滤小区域
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255),-1)
    
    cv.imshow('Tracking', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv.destroyAllWindows()