"""
Brief demo:
 - Create a small 3D cube and its vertices
 - Project the cube into a simulated camera image to obtain image points (simulate detection)
 - Use OpenCV's solvePnP to recover pose and compute distance
 - Visualize original projection and reprojected points with matplotlib

Run: python PNP.py
Requires: numpy, opencv-python (or opencv-contrib-python), matplotlib
"""

import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_cube(size=0.1):
    """返回立方体的 8 个顶点（单位：米），中心在原点。"""
    s = size / 2.0
    pts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float32)
    return pts


def camera_matrix(fx, fy, cx, cy):
    """构造相机内参矩阵 K。"""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def draw_cube_on_image(img, imgpts, color=(0, 255, 0), thickness=2):
    """在图像上画出立方体的线框（输入为 8 个 2D 点）。"""
    imgpts = imgpts.reshape(-1, 2).astype(int)
    # 底面
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)
    # 顶面
    for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)
    # 竖直边
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, thickness)


def main():
    # --- 配置 ---
    img_w, img_h = 800, 600
    fx = fy = 800.0
    cx, cy = img_w / 2.0, img_h / 2.0
    K = camera_matrix(fx, fy, cx, cy)

    # 立方体尺寸（米）
    cube = create_cube(size=0.2)

    # 定义真实的物体位姿（相机坐标系下），后续用 PnP 恢复
    # 平移（x, y, z）——物体位于相机前方
    t_true = np.array([[0.05], [0.0], [1.5]], dtype=np.float64)  # 1.5m 前方
    # 旋转：对 X、Y、Z 轴施加小角度
    rx, ry, rz = math.radians(10), math.radians(-5), math.radians(15)
    Rx = cv2.Rodrigues(np.array([rx, 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, ry, 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
    R_true = Rz @ Ry @ Rx
    rvec_true = cv2.Rodrigues(R_true)[0]

    # 将 3D 点投影到图像平面
    imgpts_true, _ = cv2.projectPoints(cube, rvec_true, t_true, K, distCoeffs=None)

    # 添加高斯噪声，模拟检测误差（像素级）
    noise_std_px = 1.0
    imgpts_noisy = imgpts_true.reshape(-1, 2) + np.random.normal(0, noise_std_px, (8, 2))

    # 使用 solvePnP 求解位姿
    success, rvec_est, tvec_est = cv2.solvePnP(cube, imgpts_noisy, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        print('solvePnP failed')
        sys.exit(1)

    # 重投影用于比较
    reproj, _ = cv2.projectPoints(cube, rvec_est, tvec_est, K, None)

    # 计算距离（相机中心到物体原点）
    dist_true = np.linalg.norm(t_true)
    dist_est = np.linalg.norm(tvec_est)

    # 打印结果摘要
    print('--- PnP distance demo ---')
    print(f'True translation (m): {t_true.ravel()}, True distance: {dist_true:.3f} m')
    print(f'Estimated translation (m): {tvec_est.ravel()}, Estimated distance: {dist_est:.3f} m')
    print('rvec_true:', rvec_true.ravel())
    print('rvec_est :', rvec_est.ravel())

    # 可视化：左为真实投影与检测点（红点），右为重投影（估计）
    img_orig = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    img_reproj = img_orig.copy()

    # 画出真实投影（绿色）与带噪声检测（红点）
    draw_cube_on_image(img_orig, imgpts_true)
    for p in imgpts_noisy:
        cv2.circle(img_orig, tuple(p.astype(int)), 4, (0, 0, 255), -1)

    # 画出重投影（蓝色）
    draw_cube_on_image(img_reproj, reproj, color=(255, 0, 0))

    # 使用 Matplotlib 显示两张图像和 3D 示意
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title('True projection + detections (red dots)')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(img_reproj, cv2.COLOR_BGR2RGB))
    ax2.set_title('Reprojection (blue lines)')
    ax2.axis('off')

    # 3D 示意：立方体与相机位置（相机在原点）
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    cube_cam = (R_true @ cube.T).T + t_true.ravel()
    ax3.scatter(cube_cam[:, 0], cube_cam[:, 1], cube_cam[:, 2], c='g')
    ax3.scatter([0], [0], [0], c='r', marker='^', s=60)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('In camera coords: cube (green) and camera (red)')
    ax3.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()

    # 简要说明 PnP 流程（中文注释）
    print('\nPNP 简要流程：')
    print('1）已知物体上若干已知 3D 点（物体坐标系）及对应的 2D 图像点（像素）')
    print('2）给定相机内参矩阵 K，使用 solvePnP 求解 rvec 和 tvec（物体相对相机的位姿）')
    print('3）tvec 的范数给出相机到物体原点的距离；位姿可用于重投影验证')


if __name__ == "__main__":
    main()

