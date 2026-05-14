import os
import csv
import cv2
import numpy as np
from glob import glob


def merge_rects_by_center_distance(rects, distance_threshold=10):
    """
    根据矩形中心之间的欧氏距离合并矩形。
    """
    if not rects:
        return []

    n = len(rects)
    centers = []
    for (x, y, w, h) in rects:
        cx = x + w / 2.0
        cy = y + h / 2.0
        centers.append((cx, cy))

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist <= distance_threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(rects[i])

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            x_min = min(r[0] for r in group)
            y_min = min(r[1] for r in group)
            x_max = max(r[0] + r[2] for r in group)
            y_max = max(r[1] + r[3] for r in group)
            merged.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return merged


def wlrdc_detect(
    image_path,
    output_dir,
    target_size=3,
    pad_size=3,
    weight_sigma=10.0,
    thresh_ratio=0.5,
    min_area=2,
    save_enhanced=True,
    analyze=True,
    save_marked=True,
    merge_distance=10
):
    """
    WLRDC: Weighted Local Ratio-Difference Contrast + 目标分析
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"无法读取图像: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    img = img_raw.astype(np.float32)
    eps = 1e-6
    h, w = img.shape

    kernel = np.ones((target_size, target_size), np.float32) / (target_size ** 2)
    center_mean = cv2.filter2D(img, -1, kernel)

    offsets = [(0, -1), (0, 1), (-1, 0), (1, 0),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    saliency = np.full((h, w), np.inf, dtype=np.float32)
    for dx, dy in offsets:
        shifted = np.roll(img, shift=(dy * pad_size, dx * pad_size), axis=(0, 1))
        mask = np.ones_like(img, dtype=bool)
        if dx > 0:
            mask[:, :dx * pad_size] = False
        elif dx < 0:
            mask[:, dx * pad_size:] = False
        if dy > 0:
            mask[:dy * pad_size, :] = False
        elif dy < 0:
            mask[dy * pad_size:, :] = False

        neighbor_mean = cv2.filter2D(shifted, -1, kernel)
        neighbor_mean[~mask] = np.nan

        contrast = np.maximum(0, (center_mean - neighbor_mean) * (center_mean / (neighbor_mean + eps)))
        saliency = np.fmin(saliency, contrast)

    saliency = np.nan_to_num(saliency, nan=0.0)

    bg_size = target_size + 2 * pad_size
    bg_std = cv2.blur(img ** 2, (bg_size, bg_size)) - cv2.blur(img, (bg_size, bg_size)) ** 2
    bg_std = np.sqrt(np.maximum(bg_std, 0))
    weight = 1.0 / (1.0 + bg_std / weight_sigma)
    saliency = saliency * weight

    vmax = saliency.max()
    if vmax > 0:
        _, binary = cv2.threshold(saliency, vmax * thresh_ratio, 255, cv2.THRESH_BINARY)
    else:
        binary = np.zeros_like(saliency, dtype=np.uint8)
    binary = binary.astype(np.uint8)

    # 面积滤波
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    binary = clean  # 直接用 clean 作为最终二值图

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), binary)
    if save_enhanced:
        norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_wlrdc.png"), norm)

    # ========== 新增目标分析模块 ==========
    targets_info = []
    if analyze:
        # 从已做面积滤波的 binary 中提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in contours]

        # 基于中心距离合并
        merged_rects = merge_rects_by_center_distance(rects, merge_distance)

        # 标记图像
        marked_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

        for idx, (x, y, w, h) in enumerate(merged_rects, start=1):
            roi_img = img_raw[y:y+h, x:x+w]
            roi_sal = saliency[y:y+h, x:x+w]

            mean_intensity = np.mean(roi_img) if roi_img.size > 0 else 0
            max_intensity = np.max(roi_img) if roi_img.size > 0 else 0
            saliency_mean = np.mean(roi_sal) if roi_sal.size > 0 else 0
            area = w * h
            cx = x + w / 2.0
            cy = y + h / 2.0

            target = {
                "image": base_name,
                "id": idx,
                "area": area,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "mean_intensity": round(mean_intensity, 2),
                "max_intensity": round(max_intensity, 2),
                "saliency_mean": round(saliency_mean, 2),
            }
            targets_info.append(target)

            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(marked_img, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if save_marked and targets_info:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_marked.png"), marked_img)

        csv_file = os.path.join(output_dir, f"{base_name}_targets.csv")
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ["image", "id", "area", "cx", "cy",
                          "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                          "mean_intensity", "max_intensity", "saliency_mean"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in targets_info:
                writer.writerow(t)

        print(f"WLRDC 已处理: {base_name}, 检测到 {len(targets_info)} 个目标 "
              f"(中心距离≤{merge_distance}合并, 面积≥{min_area})")
    else:
        print(f"WLRDC 已处理: {base_name}")
    # ========== 分析模块结束 ==========


def batch_wlrdc(input_folder, output_folder, **kwargs):
    os.makedirs(output_folder, exist_ok=True)
    files = []
    for ext in ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        files.extend(glob(os.path.join(input_folder, ext)))
    if not files:
        print(f"在 {input_folder} 中未找到图像文件。")
        return
    print(f"共 {len(files)} 张图像，开始 WLRDC 批量处理...")
    for f in files:
        wlrdc_detect(f, output_folder, **kwargs)
    print("批量处理完成。")


if __name__ == "__main__":
    batch_wlrdc(
        input_folder="./images",
        output_folder="./wlrdc_results",
        target_size=7,
        pad_size=5,
        weight_sigma=10.0,
        thresh_ratio=0.5,
        min_area=2,
        save_enhanced=True,
        analyze=True,
        save_marked=True,
        merge_distance=10
    )