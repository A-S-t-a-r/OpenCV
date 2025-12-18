1. 视频读取与播放
   1. 视频文件: cv2.VideoCapture('example.mp4')
   2. 摄像头: cv2.VideoCapture(0)
2. 帧率控制
   1. 基本操作: gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   2. 保存: 
        ```
        # 获取视频的帧率和尺寸
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 创建 VideoWriter 对象，保存处理后的视频
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
        ```