"""特征匹配示例脚本

流程：
1. 加载模板图片 `image/OpenCV三周入门手册/feature_map.png`
2. 对模板应用一个随机仿射变换生成测试图
3. 用 ORB 提取特征（关键点 + 描述子）
4. 用 BFMatcher (Hamming) 和 FLANN (LSH) 分别做 KNN 匹配并应用 Lowe 的 ratio test
5. 基于匹配点用 RANSAC 估计仿射并筛除外点（鲁棒估计）
6. 展示并保存 BF 与 FLANN 的匹配效果（上下两图）

注意：脚本使用 4 个空格缩进。
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
import time


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到测试图片: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图片: {path}")
    return img


def random_affine(img, scale_range=(0.6, 0.95), allow_rotation=False, max_angle=0):
    """生成一个缩小 + 位移的仿射变换，保证目标图像完全位于画面内部。

    - scale_range: (min_s, max_s) 缩放比例区间（小于1 为缩小）
    - allow_rotation: 是否允许额外旋转（默认为 False）
    - max_angle: 当 allow_rotation 为 True 时，允许的最大旋转角度（度）
    返回 (warped, M)
    """
    h, w = img.shape[:2]
    # 只做缩放（缩小）和位移，保证缩放后的图像能完全放入原始画布
    s = float(np.random.uniform(scale_range[0], scale_range[1]))
    new_w = int(w * s)
    new_h = int(h * s)

    # 确保至少为 1
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 随机位移，使得 [tx, tx+new_w) 在 [0, w) 内
    max_tx = w - new_w
    max_ty = h - new_h
    if max_tx < 0 or max_ty < 0:
        # fallback: 不缩放
        s = 1.0
        new_w, new_h = w, h
        max_tx, max_ty = 0, 0

    tx = int(np.random.randint(0, max_tx + 1)) if max_tx > 0 else 0
    ty = int(np.random.randint(0, max_ty + 1)) if max_ty > 0 else 0

    # 初始为仅缩放和平移的仿射矩阵（关于原点）
    M = np.array([[s, 0, tx],
                  [0, s, ty]], dtype=np.float32)

    if allow_rotation and max_angle > 0:
        angle = np.random.uniform(-max_angle, max_angle)
        # 旋转围绕缩放后图像中心（cx, cy）
        cx = tx + new_w / 2.0
        cy = ty + new_h / 2.0
        R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        # 将 R (2x3) 与缩放/平移 M (2x3) 组合：先缩放平移，再旋转 => R * [M; 0 0 1]
        # 扩展为 3x3 矩阵相乘
        M_ext = np.vstack([M, [0, 0, 1]])
        R_ext = np.vstack([R, [0, 0, 1]])
        M_comb = R_ext.dot(M_ext)
        M = M_comb[:2, :]

    # 使用常数边界，不做平铺/反射
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, M


def knn_ratio_test(matcher, des1, des2, ratio=0.75):
    """返回满足 Lowe ratio 的 match 列表"""
    if des1 is None or des2 is None:
        return []
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        # 某些 matcher 对 uint8/其他类型敏感
        des1_f = np.asarray(des1, np.uint8) if des1 is not None else None
        des2_f = np.asarray(des2, np.uint8) if des2 is not None else None
        matches = matcher.knnMatch(des1_f, des2_f, k=2)

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def ransac_filter(kp1, kp2, matches, ransac_thresh=3.0):
    """使用 RANSAC 估计仿射并返回内点 matches 和 mask"""
    if len(matches) < 3:
        return matches, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    # estimateAffinePartial2D 更适合仿射
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if mask is None:
        return matches, None
    mask = mask.ravel().tolist()
    inliers = [m for m, v in zip(matches, mask) if v]
    return inliers, mask, M


def draw_matches(img1, kp1, img2, kp2, matches, mask=None, title=None):
    draw_params = dict(
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    if mask is not None:
        draw_params['matchColor'] = (0, 255, 0)
        draw_params['singlePointColor'] = (255, 0, 0)
        draw_params['matchesMask'] = mask
    img_draw = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    if title:
        # 在图上加标题（PIL/Matplotlib 显示上层会再显示标题）
        cv2.putText(img_draw, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_draw


def evaluate_params(img, img2, orb_params, ratio, ransac_thresh, flann_params):
    """用给定参数评估 BF 与 FLANN 的 inlier 数量，返回统计和绘图数据"""
    try:
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(**orb_params)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # BF
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        good_bf = knn_ratio_test(bf, des1, des2, ratio=ratio)
        inliers_bf, mask_bf, M_bf = ransac_filter(kp1, kp2, good_bf, ransac_thresh=ransac_thresh)

        # FLANN
        try:
            flann = cv2.FlannBasedMatcher(flann_params, dict(checks=50))
            des1_f = np.asarray(des1, np.uint8) if des1 is not None else None
            des2_f = np.asarray(des2, np.uint8) if des2 is not None else None
            good_flann = knn_ratio_test(flann, des1_f, des2_f, ratio=ratio)
            inliers_flann, mask_flann, M_flann = ransac_filter(kp1, kp2, good_flann, ransac_thresh=ransac_thresh)
        except Exception:
            good_flann = []
            inliers_flann = []
            mask_flann = None
            M_flann = None

        return dict(kp1=len(kp1), kp2=len(kp2),
                    kp1_list=kp1, kp2_list=kp2,
                    bf_matches=len(good_bf), bf_inliers=len(inliers_bf), bf_inliers_list=inliers_bf, bf_mask=mask_bf,
                    flann_matches=len(good_flann), flann_inliers=len(inliers_flann), flann_inliers_list=inliers_flann, flann_mask=mask_flann)
    except Exception as e:
        print(f"evaluate_params error: {e}")
        return dict(kp1=0, kp2=0, kp1_list=None, kp2_list=None,
                    bf_matches=0, bf_inliers=0, bf_inliers_list=[], bf_mask=None,
                    flann_matches=0, flann_inliers=0, flann_inliers_list=[], flann_mask=None)


def grid_search(img, img2, max_combos=100, verbose=True):
    # 定义参数网格（可根据需要扩展）
    orb_nfeatures = [1000, 1500]
    orb_scaleFactor = [1.2, 1.3]
    orb_nlevels = [8, 12]
    orb_fastThreshold = [5, 20]
    # 新增描述子/关键点相关参数
    orb_WTA_K = [2, 3]
    orb_patchSize = [16, 31]
    orb_edgeThreshold = [5, 15]
    orb_scoreType = [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]

    ratio_list = [0.6, 0.75, 0.85]
    ransac_list = [2.0, 3.0, 4.0]

    flann_table_number = [6, 12]
    flann_key_size = [12, 20]
    flann_multi_probe = [1, 2]

    combos = []
    for of, sf, nl, ft, w, ps, et, st, r, rt in itertools.product(
            orb_nfeatures, orb_scaleFactor, orb_nlevels, orb_fastThreshold,
            orb_WTA_K, orb_patchSize, orb_edgeThreshold, orb_scoreType,
            ratio_list, ransac_list):
        for tn, ks, mp in itertools.product(flann_table_number, flann_key_size, flann_multi_probe):
            orb_params = dict(nfeatures=of, scaleFactor=sf, nlevels=nl, fastThreshold=ft,
                              WTA_K=w, patchSize=ps, edgeThreshold=et, scoreType=st)
            flann_params = dict(algorithm=6, table_number=tn, key_size=ks, multi_probe_level=mp)
            combos.append((orb_params, r, rt, flann_params))

    total_combos = len(combos)
    if total_combos == 0:
        return None, None

    # 若组合数超过 max_combos，则使用随机采样（可重复，由 np.random.seed 控制）
    if total_combos > max_combos:
        idxs = np.random.choice(total_combos, size=max_combos, replace=False)
        selected = [combos[i] for i in idxs]
    else:
        selected = combos

    total = len(selected)
    if verbose:
        print(f"Grid search total combos (all): {total_combos}, selected for run: {total}")

    best_bf = None
    best_flann = None
    start = time.time()
    for i, (orb_params, ratio, ransac_thresh, flann_params) in enumerate(selected):
        res = evaluate_params(img, img2, orb_params, ratio, ransac_thresh, flann_params)
        if verbose and (i % 10 == 0):
            print(f"[{i}/{total}] orb={orb_params['nfeatures']},WTA_K={orb_params['WTA_K']},ratio={ratio},ransac={ransac_thresh} -> BF inliers={res['bf_inliers']}, FLANN inliers={res['flann_inliers']}")

        if best_bf is None or res.get('bf_inliers', 0) > best_bf['res'].get('bf_inliers', 0):
            best_bf = dict(res=res, orb_params=orb_params, ratio=ratio, ransac_thresh=ransac_thresh, flann_params=flann_params)
        if best_flann is None or res.get('flann_inliers', 0) > best_flann['res'].get('flann_inliers', 0):
            best_flann = dict(res=res, orb_params=orb_params, ratio=ratio, ransac_thresh=ransac_thresh, flann_params=flann_params)

    elapsed = time.time() - start
    if verbose:
        print(f"Grid search done in {elapsed:.1f}s. Best BF inliers={best_bf['res']['bf_inliers']}, Best FLANN inliers={best_flann['res']['flann_inliers']}")
    return best_bf, best_flann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', action='store_true', help='run grid search to optimize params')
    parser.add_argument('--max-combos', type=int, default=100, help='max parameter combos to try')
    parser.add_argument('--seed', type=int, default=None, help='random seed for repeatability')
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(root, 'Data', 'feature_map.png')
    img = load_image(img_path)

    if args.seed is not None:
        np.random.seed(args.seed)

    # 生成随机仿射图
    img2, M = random_affine(img)
    print(f"Applied random affine transform:\n{M}")

    if args.grid:
        best_bf, best_flann = grid_search(img, img2, max_combos=args.max_combos)

        # 把最好结果的匹配图保存下来（使用 evaluate 返回的 kp 列表，避免重新检测导致索引不匹配）
        for label, best in [('BF', best_bf), ('FLANN', best_flann)]:
            res = best['res']
            if label == 'BF':
                matches = res['bf_inliers_list']
                mask = [1] * len(matches) if matches else None
            else:
                matches = res['flann_inliers_list']
                mask = [1] * len(matches) if matches else None

            kp1_list = res.get('kp1_list')
            kp2_list = res.get('kp2_list')
            if kp1_list is None or kp2_list is None:
                # 退化为重新检测（不推荐）
                orb_tmp = cv2.ORB_create(**best['orb_params'])
                kp1_list, _ = orb_tmp.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                kp2_list, _ = orb_tmp.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

            title = f"{label} best inliers={best['res'][f'{label.lower()}_inliers']}"
            img_draw = draw_matches(img, kp1_list, img2, kp2_list, matches, mask=mask, title=title)

            out_dir = os.path.join(root, 'image')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'feature_matches_{label.lower()}_best.png')
            cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
            print(f"Saved {label} best to {out_path}")
        return

    # 以下为默认单次评估（保留之前的行为）
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB 特征
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    print(f"ORB: img1 kp={len(kp1)}, img2 kp={len(kp2)}")

    # BFMatcher (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good_bf = knn_ratio_test(bf, des1, des2, ratio=0.75)
    print(f"BFMatcher: after ratio test {len(good_bf)} matches")
    inliers_bf, mask_bf, _ = ransac_filter(kp1, kp2, good_bf, ransac_thresh=3.0)
    print(f"BFMatcher: after RANSAC {len(inliers_bf)} inliers")

    # FLANN (LSH for ORB)
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # FLANN 对二进制描述子有时需要 uint8
        des1_f = np.asarray(des1, np.uint8) if des1 is not None else None
        des2_f = np.asarray(des2, np.uint8) if des2 is not None else None
        good_flann = knn_ratio_test(flann, des1_f, des2_f, ratio=0.75)
        print(f"FLANN: after ratio test {len(good_flann)} matches")
        inliers_flann, mask_flann, _ = ransac_filter(kp1, kp2, good_flann, ransac_thresh=3.0)
        print(f"FLANN: after RANSAC {len(inliers_flann)} inliers")
    except Exception as e:
        print(f"FLANN matcher error: {e}")
        good_flann = []
        inliers_flann = []
        mask_flann = None

    # 绘制匹配（仅绘制内点）
    img_bf = draw_matches(img, kp1, img2, kp2, inliers_bf, mask=[1] * len(inliers_bf) if inliers_bf else None)
    img_flann = draw_matches(img, kp1, img2, kp2, inliers_flann, mask=[1] * len(inliers_flann) if inliers_flann else None)

    # 显示上( BF ) 下( FLANN ) 两图
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    axes[0].imshow(img_bf)
    axes[0].set_title(f"BFMatcher (inliers: {len(inliers_bf)})")
    axes[0].axis('off')

    axes[1].imshow(img_flann)
    axes[1].set_title(f"FLANN (inliers: {len(inliers_flann)})")
    axes[1].axis('off')

    plt.tight_layout()

    # 保存对比图
    out_dir = os.path.join(root, 'Data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'feature_matches_compare.png')
    fig.savefig(out_path)
    print(f"Saved comparison image to: {out_path}")

    plt.show()


if __name__ == '__main__':
    main()