import cv2

cap = cv2.VideoCapture('Data/Example.MP4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 将帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 显示灰度帧
    cv2.imshow('Gray Video', gray_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()