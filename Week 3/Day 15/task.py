from typing import Tuple
import cv2
import numpy as np
import os

def draw_harris(img:np.ndarray, gray: np.ndarray)-> np.ndarray:
    dst = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.04)
    dst = cv2.dilate(dst,None)
    #阈值：取最大值的1%
    thresh = 0.01 * dst.max()
    out = img.copy()
    ys,xs = np.where(dst > thresh)
    for (x, y) in zip(xs, ys):
        cv2.circle(out, (x, y), 3, (0, 0, 255), 1) # 红色
    return out

def draw_shitomasi(img: np.ndarray,gray: np.ndarray,max_corners=200)-> np.ndarray:
    corners = cv2.goodFeaturesToTrack(gray,maxCorners=max_corners,qualityLevel=0.01,minDistance=10)
    out = img.copy()
    if corners is not None:
        for c in corners:
            x, y = c.ravel().astype(int)
            cv2.circle(out, (x, y), 4, (0, 255, 0), 1) # 绿色
    return out

def draw_fast(img: np.ndarray, gray: np.ndarray)-> np.ndarray:
    fast = cv2.FastFeatureDetector_create(threshold=25,nonmaxSuppression=True)
    kps = fast.detect(gray, None)
    out = cv2.drawKeypoints(img, kps, None, color=(255, 0, 0)) # 蓝色
    return out

def make_grid(a: np.ndarray,b: np.ndarray,c: np.ndarray,d: np.ndarray)-> np.ndarray:
# 确保四张图同样大小（使用 a 的尺寸）
    h, w = a.shape[:2]

    def fit(img: np.ndarray)-> np.ndarray:
        return cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

    a, b, c, d = fit(a), fit(b), fit(c), fit(d)
    top = np.hstack((a, b))
    bot = np.hstack((c, d))
    grid = np.vstack((top, bot))
    return grid

def load_image(path: str)-> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f" 找不到或无法读取图像: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def main()-> None:
    in_path = "Data/feature_map.png"
    out_path = "Data/output.png"
    img, gray = load_image(in_path)

    harris_img = draw_harris(img, gray)
    shi_img = draw_shitomasi(img, gray)
    fast_img = draw_fast(img, gray)

    # 组成 2x2 网格：左上原图，右上 Harris，左下 Shi‐Tomasi，右下 FAST
    grid = make_grid(img, harris_img, shi_img, fast_img)

    # 在每个子图上加文字标注（中文）
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    small = 1 if grid.shape[1] > 800 else 0.7
    thickness = 2
    # 写上 4 个角落的标签
    h, w = grid.shape[:2]
    wh = w // 2
    hh = h // 2
    cv2.putText(grid, '原图', (10, 30), label_font, small, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(grid, 'Harris', (wh + 10, 30), label_font, small, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(grid, 'Shi‐Tomasi', (10, hh + 30), label_font, small, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(grid, 'FAST', (wh + 10, hh + 30), label_font, small, (255, 255, 255), thickness, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f" 已保存对比图: {out_path}")
    
if __name__ == '__main__':
    main()