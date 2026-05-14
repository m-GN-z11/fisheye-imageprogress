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


def tophat_adaptive_threshold(
    image_path,
    output_dir,
    morph_shape=cv2.MORPH_ELLIPSE,
    morph_ksize=5,
    block_size=11,
    C=2,
    save_enhanced=True,
    analyze=True,
    min_area=2,
    save_marked=True,
    merge_distance=10
):
    """
    基于 Top-Hat + 局部自适应阈值的红外小目标检测 + 目标分析
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. 形态学 Top-Hat 变换 -----
    kernel = cv2.getStructuringElement(morph_shape, (morph_ksize, morph_ksize))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # ----- 2. 局部自适应阈值 -----
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

    # ----- 3. 保存基础结果 -----
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    binary = cv2.bitwise_not(binary)  # 目标变为白色
    binary_file = os.path.join(output_dir, f"{base_name}_binary.png")
    cv2.imwrite(binary_file, binary)

    if save_enhanced:
        norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_file = os.path.join(output_dir, f"{base_name}_tophat.png")
        cv2.imwrite(enhanced_file, norm.astype(np.uint8))

    # ========== 新增加的目标分析模块 ==========
    targets_info = []
    if analyze:
        # 确保 binary 是 uint8 类型 (经过 bitwise_not 后已是)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 面积过滤
        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, w, h))

        # 中心距离合并
        merged_rects = merge_rects_by_center_distance(rects, merge_distance)

        # 绘制标记图
        marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for idx, (x, y, w, h) in enumerate(merged_rects, start=1):
            roi_img = img[y:y+h, x:x+w]
            roi_tophat = tophat[y:y+h, x:x+w]

            mean_intensity = np.mean(roi_img) if roi_img.size > 0 else 0
            max_intensity = np.max(roi_img) if roi_img.size > 0 else 0
            tophat_mean = np.mean(roi_tophat) if roi_tophat.size > 0 else 0
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
                "tophat_mean": round(tophat_mean, 2),
            }
            targets_info.append(target)

            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(marked_img, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if save_marked and targets_info:
            marked_file = os.path.join(output_dir, f"{base_name}_marked.png")
            cv2.imwrite(marked_file, marked_img)

        # 保存 CSV
        csv_file = os.path.join(output_dir, f"{base_name}_targets.csv")
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ["image", "id", "area", "cx", "cy",
                          "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                          "mean_intensity", "max_intensity", "tophat_mean"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in targets_info:
                writer.writerow(t)

        print(f"已处理: {base_name}, 检测到 {len(targets_info)} 个目标 "
              f"(中心距离≤{merge_distance}合并, 面积≥{min_area})")
    else:
        print(f"已处理: {base_name}")
    # ========== 分析模块结束 ==========


def batch_tophat_adaptive(
    input_folder,
    output_folder,
    morph_shape=cv2.MORPH_ELLIPSE,
    morph_ksize=5,
    block_size=15,
    C=2,
    save_enhanced=True,
    **kwargs
):
    """批量处理文件夹内的所有图像，支持额外分析参数"""
    os.makedirs(output_folder, exist_ok=True)

    files = []
    for ext in ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        files.extend(glob(os.path.join(input_folder, ext)))

    if not files:
        print(f"在 {input_folder} 中未找到图像文件。")
        return

    print(f"共找到 {len(files)} 张图像，开始批量处理...")
    for f in files:
        tophat_adaptive_threshold(
            f,
            output_folder,
            morph_shape=morph_shape,
            morph_ksize=morph_ksize,
            block_size=block_size,
            C=C,
            save_enhanced=save_enhanced,
            **kwargs
        )
    print("批量处理完成。")


if __name__ == "__main__":
    # ========== 参数可调区域 ==========
    SHAPE = cv2.MORPH_RECT
    KSIZE = 3
    BLOCK = 11
    CONST_C = 3
    SAVE_ENHANCE = True

    # 目标分析参数
    ANALYZE = True
    MIN_AREA = 2
    SAVE_MARKED = True
    MERGE_DIST = 10

    batch_tophat_adaptive(
        input_folder="./images",
        output_folder="./swlc_results",
        morph_shape=SHAPE,
        morph_ksize=KSIZE,
        block_size=BLOCK,
        C=CONST_C,
        save_enhanced=SAVE_ENHANCE,
        analyze=ANALYZE,
        min_area=MIN_AREA,
        save_marked=SAVE_MARKED,
        merge_distance=MERGE_DIST
    )