
"""特征描述子展示脚本

将图片 image/OpenCV三周入门手册/feature_map.png 作为测试图，按 3x2 网格展示：
第一张为原图（不做特征检测），其余依次展示 SIFT, SURF, BRIEF, AKAZE, ORB 的检测结果。

注意：SURF/BRIEF 依赖于 opencv-contrib 的 xfeatures2d，如果不可用会显示提示信息。
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到测试图片: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图片: {path}")
    return img


def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    # 返回绘制关键点的彩色图像（RGB）
    out = cv2.drawKeypoints(img, keypoints, None, color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def placeholder_text(shape, text):
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(out, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # 图片相对工作区路径
    img = cv2.imread('Data/feature_map.png')

    # 转为灰度用于检测/描述
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = []

    # 原图（不检测）
    results.append(('Original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    # 1) SIFT
    try:
        sift = cv2.SIFT_create()
        kp_sift = sift.detect(gray, None)
        print(f"SIFT: {len(kp_sift)} keypoints detected")
        img_sift = draw_keypoints(img, kp_sift)
        results.append(('SIFT', img_sift))
    except Exception as e:
        print(f"SIFT error: {e}")
        results.append(('SIFT', placeholder_text(img.shape, 'SIFT error')))

    # 2) SURF (may be unavailable) — fallback to KAZE
    try:
        surf = None
        # attempt creation in a nested try so we can catch nonfree-build errors
        try:
            if hasattr(cv2, 'xfeatures2d'):
                surf = cv2.xfeatures2d.SURF_create(400)
            else:
                # some builds might expose it differently, try getattr
                x = getattr(cv2, 'xfeatures2d', None)
                if x is not None and hasattr(x, 'SURF_create'):
                    surf = x.SURF_create(400)
        except Exception as e_surf_create:
            print(f"SURF create failed: {e_surf_create}")
            surf = None

        if surf is not None:
            kp_surf = surf.detect(gray, None)
            print(f"SURF: {len(kp_surf)} keypoints detected")
            img_surf = draw_keypoints(img, kp_surf)
            results.append(('SURF', img_surf))
        else:
            # fallback to KAZE when SURF not available or creation failed
            kaze = cv2.KAZE_create()
            kp_kaze = kaze.detect(gray, None)
            print(f"SURF not available, used KAZE: {len(kp_kaze)} keypoints detected")
            img_kaze = draw_keypoints(img, kp_kaze)
            results.append(('KAZE (SURF substitute)', img_kaze))
    except Exception as e:
        print(f"SURF/KAZE unexpected error: {e}")
        results.append(('SURF/KAZE error', placeholder_text(img.shape, 'SURF/KAZE error')))

    # 3) BRIEF (descriptor only, use FAST keypoints)
    try:
        brief = None
        if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'BriefDescriptorExtractor_create'):
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        elif hasattr(cv2, 'xfeatures2d') and getattr(cv2.xfeatures2d, 'BriefDescriptorExtractor_create', None):
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        if brief is None:
            raise AttributeError('BRIEF not available')

        fast = cv2.FastFeatureDetector_create()
        kp_fast = fast.detect(gray, None)
        kp_fast, des = brief.compute(gray, kp_fast)
        n_desc = 0 if des is None else des.shape[0]
        print(f"BRIEF: {len(kp_fast)} keypoints detected, descriptors: {n_desc}")
        img_brief = draw_keypoints(img, kp_fast)
        results.append(('BRIEF', img_brief))
    except Exception as e:
        print(f"BRIEF error or not available: {e}")
        results.append(('BRIEF', placeholder_text(img.shape, 'BRIEF not available')))

    # 4) AKAZE
    try:
        akaze = cv2.AKAZE_create()
        kp_akaze = akaze.detect(gray, None)
        print(f"AKAZE: {len(kp_akaze)} keypoints detected")
        img_akaze = draw_keypoints(img, kp_akaze)
        results.append(('AKAZE', img_akaze))
    except Exception as e:
        print(f"AKAZE error: {e}")
        results.append(('AKAZE', placeholder_text(img.shape, 'AKAZE error')))

    # 5) ORB
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kp_orb = orb.detect(gray, None)
        print(f"ORB: {len(kp_orb)} keypoints detected")
        img_orb = draw_keypoints(img, kp_orb)
        results.append(('ORB', img_orb))
    except Exception as e:
        print(f"ORB error: {e}")
        results.append(('ORB', placeholder_text(img.shape, 'ORB error')))

    # 显示 3x2 网格
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    axes = axes.flatten()
    for ax, (title, im) in zip(axes, results):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    # 如果 entries 少于6，则填充空白
    for i in range(len(results), 6):
        axes[i].imshow(np.zeros_like(results[0][1]))
        axes[i].set_title('')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
