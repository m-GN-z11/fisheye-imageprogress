import os
import csv
import cv2
import numpy as np
from glob import glob


def merge_rects_by_center_distance(rects, distance_threshold=10):
    """
    根据矩形中心之间的欧氏距离合并矩形。
    将所有中心距离 ≤ distance_threshold 的矩形归为一组，
    每组生成一个能覆盖该组所有矩形的最小外接矩形。
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


def widlcm_detect(
    image_path,
    output_dir,
    inner_size=3,
    mid_size=7,
    outer_size=11,
    weight_inner=0.6,
    thresh_ratio=0.5,
    min_area=2,
    save_enhanced=True,
    analyze=True,
    save_marked=True,
    merge_distance=10
):
    """
    WIDLCM: 加权改进双局部对比度 + 目标分析
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"无法读取图像: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    img = img_raw.astype(np.float32)

    # 三层均值
    inner_mean = cv2.blur(img, (inner_size, inner_size))
    mid_mean = cv2.blur(img, (mid_size, mid_size))
    outer_mean = cv2.blur(img, (outer_size, outer_size))

    inner_area = inner_size ** 2
    mid_area = mid_size ** 2
    outer_area = outer_size ** 2
    eps = 1e-6

    mid_ring = (mid_mean * mid_area - inner_mean * inner_area) / (mid_area - inner_area + eps)
    outer_ring = (outer_mean * outer_area - mid_mean * mid_area) / (outer_area - mid_area + eps)

    # 两个对比度分量
    C1 = (inner_mean - mid_ring) * (inner_mean / (mid_ring + eps))
    C2 = (inner_mean - outer_ring) * (inner_mean / (outer_ring + eps))

    # 加权融合
    saliency = weight_inner * C1 + (1 - weight_inner) * C2
    saliency = np.maximum(saliency, 0)

    # 二值化
    vmax = saliency.max()
    if vmax > 0:
        _, binary = cv2.threshold(saliency, vmax * thresh_ratio, 255, cv2.THRESH_BINARY)
    else:
        binary = np.zeros_like(saliency, dtype=np.uint8)

    # 面积滤波（连通域清理）
    binary = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    binary = clean

    # 保存基础结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), binary)
    if save_enhanced:
        norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_widlcm.png"), norm)

    # ========== 新增加的目标分析模块 ==========
    targets_info = []
    if analyze:
        # binary 已经过面积滤波，直接提取轮廓即可
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集矩形（不再需要面积过滤，可直接保留所有轮廓）
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))

        # 中心距离合并
        merged_rects = merge_rects_by_center_distance(rects, merge_distance)

        # 绘制标记图
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

        # 保存 CSV
        csv_file = os.path.join(output_dir, f"{base_name}_targets.csv")
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ["image", "id", "area", "cx", "cy",
                          "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                          "mean_intensity", "max_intensity", "saliency_mean"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in targets_info:
                writer.writerow(t)

        print(f"WIDLCM 已处理: {base_name}, 检测到 {len(targets_info)} 个目标 "
              f"(中心距离≤{merge_distance}合并, 面积≥{min_area})")
    else:
        print(f"WIDLCM 已处理: {base_name}")
    # ========== 分析模块结束 ==========


def batch_widlcm(input_folder, output_folder, **kwargs):
    os.makedirs(output_folder, exist_ok=True)
    files = []
    for ext in ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        files.extend(glob(os.path.join(input_folder, ext)))
    if not files:
        print(f"在 {input_folder} 中未找到图像文件。")
        return
    print(f"共 {len(files)} 张图像，开始 WIDLCM 批量处理...")
    for f in files:
        widlcm_detect(f, output_folder, **kwargs)
    print("批量处理完成。")


if __name__ == "__main__":
    batch_widlcm(
        input_folder="./images",
        output_folder="./widlcm_results",
        inner_size=3,
        mid_size=7,
        outer_size=11,
        weight_inner=0.6,
        thresh_ratio=0.5,
        min_area=2,
        save_enhanced=True,
        analyze=True,
        save_marked=True,
        merge_distance=10
    )