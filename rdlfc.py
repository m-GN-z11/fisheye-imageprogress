import os
import csv
import cv2
import numpy as np
from glob import glob


def rdlfc_detect(
    image_path,
    output_dir,
    target_size=3,
    mid_size=7,
    outer_size=11,
    thresh_ratio=0.5,
    min_area=2,
    save_enhanced=True,
    analyze=True,          # 新增：是否进行目标分析
    save_marked=True       # 新增：是否保存标记图像
):
    """
    RDLFC: Ratio-Difference Local Feature Contrast
    三层窗口 + 比值-差分乘积对比度

    参数:
        target_size : 目标区域边长 (奇数)
        mid_size    : 中间保护环边长 (奇数, > target_size)
        outer_size  : 背景外环边长 (奇数, > mid_size)
        thresh_ratio: 阈值 = max(saliency) * ratio
        min_area    : 连通域最小面积过滤
        analyze     : 是否提取目标属性并保存 CSV
        save_marked : 是否保存带编号的标记图像
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    eps = 1e-6

    # 各层均值
    inner_mean = cv2.blur(img, (target_size, target_size))
    mid_mean = cv2.blur(img, (mid_size, mid_size))
    outer_mean = cv2.blur(img, (outer_size, outer_size))

    # 计算环域均值
    inner_area = target_size ** 2
    mid_area = mid_size ** 2
    outer_area = outer_size ** 2

    mid_ring = (mid_mean * mid_area - inner_mean * inner_area) / (mid_area - inner_area + eps)
    outer_ring = (outer_mean * outer_area - mid_mean * mid_area) / (outer_area - mid_area + eps)

    # 两个对比度分量
    C1 = np.maximum(0, (inner_mean - mid_ring) * (inner_mean / (mid_ring + eps)))
    C2 = np.maximum(0, (inner_mean - outer_ring) * (inner_mean / (outer_ring + eps)))

    # 乘积融合（更强调两者同时显著）
    saliency = C1 * C2

    # 二值化
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

    # 保存基础结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), clean)
    if save_enhanced:
        norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_rdlfc.png"), norm)

    # ----- 目标分析（无合并） -----
    if analyze:
        # 对面积滤波后的二值图重新计算连通域，获得精确的 stats
        num_clean, labels_clean, stats_clean, _ = cv2.connectedComponentsWithStats(
            clean, connectivity=8
        )

        targets_info = []
        # 准备标记图像
        marked_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for i in range(1, num_clean):  # 跳过背景
            area = stats_clean[i, cv2.CC_STAT_AREA]
            if area < min_area:        # 理论上 clean 已过滤，但保留检查
                continue

            x = stats_clean[i, cv2.CC_STAT_LEFT]
            y = stats_clean[i, cv2.CC_STAT_TOP]
            w = stats_clean[i, cv2.CC_STAT_WIDTH]
            h = stats_clean[i, cv2.CC_STAT_HEIGHT]
            cx = x + w / 2.0
            cy = y + h / 2.0

            # 提取该目标区域内的像素（原图 + 显著图）
            mask = (labels_clean == i)
            roi_img = img[mask]
            roi_sal = saliency[mask]

            mean_intensity = np.mean(roi_img) if len(roi_img) > 0 else 0.0
            max_intensity = np.max(roi_img) if len(roi_img) > 0 else 0.0
            saliency_mean = np.mean(roi_sal) if len(roi_sal) > 0 else 0.0

            target = {
                "image": base_name,
                "id": len(targets_info) + 1,          # 从 1 开始连续编号
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

            # 绘制矩形框和编号
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(marked_img, str(target["id"]), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 保存标记图像
        if save_marked and targets_info:
            marked_file = os.path.join(output_dir, f"{base_name}_marked.png")
            cv2.imwrite(marked_file, marked_img)

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

        print(f"RDLFC 已处理: {base_name}, 检测到 {len(targets_info)} 个目标 (面积≥{min_area})")
    else:
        print(f"RDLFC 已处理: {base_name}")


def batch_rdlfc(input_folder, output_folder, **kwargs):
    os.makedirs(output_folder, exist_ok=True)
    files = []
    for ext in ('*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        files.extend(glob(os.path.join(input_folder, ext)))
    if not files:
        print(f"在 {input_folder} 中未找到图像文件。")
        return
    print(f"共 {len(files)} 张图像，开始 RDLFC 批量处理...")
    for f in files:
        rdlfc_detect(f, output_folder, **kwargs)
    print("批量处理完成。")


if __name__ == "__main__":
    batch_rdlfc(
        input_folder="./images",
        output_folder="./rdlfc_results",
        target_size=3,
        mid_size=7,
        outer_size=11,
        thresh_ratio=0.4,
        min_area=3,
        save_enhanced=True,
        analyze=True,       # 开启分析
        save_marked=True    # 保存标记图像
    )