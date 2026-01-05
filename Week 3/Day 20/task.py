# -*- coding: utf-8 -*-
"""
双目视觉示例：从左右摄像头视频生成视差图并保存为视频（默认把帧缩小2倍以降低计算量）
使用方法:
    python 双目视觉.py --left image/camera_left.mp4 --right image/camera_right.mp4 --output disparity.mp4

输出为带伪彩色的视差可视化视频。
"""

import argparse
import cv2
import numpy as np
import sys


def make_stereo_sgbm(min_disp=0, num_disp=128, block_size=5, channels=1):
    # num_disp 必须是 16 的倍数
    num_disp = max(16, (num_disp + 15) // 16 * 16)
    if num_disp <= 0:
        num_disp = 16

    P1 = 8 * channels * block_size * block_size
    P2 = 32 * channels * block_size * block_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
    )
    return stereo


def normalize_disp_for_display(disp):
    # 输入 disp: int16 或 float32 (16倍 disparity)，需要归一化到 0-255 并转换为 uint8
    disp_float = disp.astype(np.float32) / 16.0  # StereoSGBM 输出乘以16
    # 将无效值（小于等于 0）设为 0
    disp_float[disp_float < 0] = 0
    disp_norm = cv2.normalize(disp_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)
    return disp_uint8


def main():
    parser = argparse.ArgumentParser(description='Stereo SGBM disparity video generator')
    parser.add_argument('--left', type=str, default='D:/杂七杂八/大一上/算法组/OpenCV 第三周/image/camera_left.mp4', help='Left video path')
    parser.add_argument('--right', type=str, default="D:/杂七杂八/大一上/算法组/OpenCV 第三周/image/camera_right.mp4", help='Right video path')
    parser.add_argument('--output', type=str, default='Data/disparity_output.mp4', help='Output disparity video path')
    parser.add_argument('--scale', type=float, default=1, help='Resize scale (e.g., 0.5 means half size)')
    parser.add_argument('--numdisp', type=int, default=128, help='numDisparities (will be rounded up to multiple of 16)')
    parser.add_argument('--block', type=int, default=9, help='blockSize (odd number)')
    parser.add_argument('--show', action='store_true', help='Show live window')
    args = parser.parse_args()

    capL = cv2.VideoCapture(args.left)
    capR = cv2.VideoCapture(args.right)

    if not capL.isOpened():
        print('无法打开左视频:', args.left)
        sys.exit(1)
    if not capR.isOpened():
        print('无法打开右视频:', args.right)
        sys.exit(1)

    # 获取帧率、尺寸等信息
    fpsL = capL.get(cv2.CAP_PROP_FPS) or 25.0
    fpsR = capR.get(cv2.CAP_PROP_FPS) or 25.0
    fps = min(fpsL, fpsR)

    w = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 缩小尺寸
    out_w = int(w * args.scale)
    out_h = int(h * args.scale)

    # 确保宽度为16的倍数（便于numDisparities设置），不过这不是严格必要
    # 初始化 StereoSGBM
    channels = 1
    stereo = make_stereo_sgbm(min_disp=0, num_disp=args.numdisp, block_size=args.block, channels=channels)

    # 输出视频写入器（使用伪彩色 BGR）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    print('开始处理：')
    print(' 左: ', args.left)
    print(' 右: ', args.right)
    print(' 输出: ', args.output)
    print(f' 缩放: {args.scale} -> {out_w}x{out_h}, fps: {fps}')

    frame_idx = 0
    try:
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()

            if not retL or not retR:
                print('任意一侧视频结束，停止处理')
                break

            # 缩放
            frameL = cv2.resize(frameL, (out_w, out_h), interpolation=cv2.INTER_AREA)
            frameR = cv2.resize(frameR, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # 转灰度
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            # 计算视差 (输出为 int16，单位为 disparity*16)
            disp = stereo.compute(grayL, grayR)

            disp8 = normalize_disp_for_display(disp)

            # 伪彩色增强可视化
            disp_color = cv2.applyColorMap(disp8, cv2.COLORMAP_JET)

            # 写入输出视频
            out.write(disp_color)

            if args.show:
                # 同时显示左右和视差
                top = np.hstack((frameL, frameR))
                bottom = np.hstack((cv2.cvtColor(disp8, cv2.COLOR_GRAY2BGR), disp_color))
                vis = np.vstack((top, bottom))
                cv2.imshow('Left | Right -- Disparity(gray) | Disparity(color)', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('收到退出信号')
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f' 已处理帧: {frame_idx}')

    finally:
        capL.release()
        capR.release()
        out.release()
        cv2.destroyAllWindows()
        print('处理完成，已释放资源')


if __name__ == '__main__':
    main()