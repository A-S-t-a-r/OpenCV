import cv2

cap = cv2.VideoCapture('Data/Example.MP4')

# 获取视频的帧率和尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，保存处理后的视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 将帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 将灰度帧写入输出视频
    out.write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))
    
    # 显示灰度帧
    cv2.imshow('Gray Video', gray_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()