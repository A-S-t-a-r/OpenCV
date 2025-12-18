import cv2 as cv
import time

#读取mp4视频文件
cap = cv.VideoCapture('Data/Example.MP4')
chars = ['','#'] #二值化：空格表示白，#表示黑

while True:
    ret,frame = cap.read()
    if not ret or frame is None:
        break

    #转灰度并缩小
    if frame.ndim == 3:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    else:
        gray = frame
    small = cv.resize(gray,(120,40))

    #转字符画
    lines = []
    for row in small:
        line = ''.join('#' if pixel > 127 else '' for pixel in row)
        lines.append(line)
    ascii_art = '\n'.join(lines)

    # 清屏并打印
    print('\033[H\033[J' + ascii_art)

    time.sleep(0.03) # 控制播放速度，约 30fps

cap.release()