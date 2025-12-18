import cv2

# 创建 VideoCapture 对象，读取视频文件
cap = cv2.VideoCapture('Data/Example.MP4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 读取视频帧
while True:
    ret, frame = cap.read()
    
    # 如果读取到最后一帧，退出循环
    if not ret:
        break
    
    # 显示当前帧
    cv2.imshow('Video', frame)
    
    # 按下 'q' 键退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()