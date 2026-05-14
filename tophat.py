import os
import cv2
import numpy as np
from glob import glob


def tophat_adaptive_threshold(
    image_path,
    output_dir,
    morph_shape=cv2.MORPH_ELLIPSE,
    morph_ksize=5,
    block_size=11,
    C=2,
    save_enhanced=True
):
    """
    基于 Top-Hat + 局部自适应阈值的红外小目标检测
    ---------------------------------------------------
    1. 白顶帽变换 (White Top-Hat): 原图 - 开运算(原图)
       用椭圆形结构元素估计背景，差分后突出亮目标。
    2. 局部自适应阈值: 对 Top-Hat 增强图进行二值化，
       局部阈值分割能有效应对画面光照不均。
    3. 输出结果包含:
       - 二值检测结果（_binary.png）
       - 归一化的 Top-Hat 增强图（_tophat.png，用于观察效果）

    参数说明:
        morph_shape : 结构元素形状 (cv2.MORPH_ELLIPSE, MORPH_RECT, MORPH_CROSS)
        morph_ksize : 结构元素尺寸 (略大于目标尺寸)
        block_size  : 自适应阈值邻域大小（奇数）
        C           : 阈值常数（均值 - C），值越大检测到的目标越少
        save_enhanced: 是否保存 Top-Hat 增强图
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # ----- 1. 形态学 Top-Hat 变换 -----
    kernel = cv2.getStructuringElement(morph_shape, (morph_ksize, morph_ksize))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # ----- 2. 局部自适应阈值 (补齐缺失的局部对比度步骤) -----
    # 确保 block_size 是奇数
    if block_size % 2 == 0:
        block_size += 1

    # 在 Top-Hat 增强图上使用高斯加权的局部自适应阈值
    binary = cv2.adaptiveThreshold(
        tophat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 或 cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # ----- 3. 保存结果 -----
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 保存二值检测结果
    binary_file = os.path.join(output_dir, f"{base_name}_binary.png")
    cv2.imwrite(binary_file, binary)

    # 保存 Top-Hat 增强图 (归一化到 0-255 便于人眼查看)
    if save_enhanced:
        norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_file = os.path.join(output_dir, f"{base_name}_tophat.png")
        cv2.imwrite(enhanced_file, norm.astype(np.uint8))

    print(f"已处理: {base_name}")


def batch_tophat_adaptive(
    input_folder,
    output_folder,
    morph_shape=cv2.MORPH_ELLIPSE,
    morph_ksize=5,
    block_size=15,
    C=2,
    save_enhanced=True
):
    """批量处理文件夹内的所有图像"""
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
            save_enhanced=save_enhanced
        )
    print("批量处理完成。")


if __name__ == "__main__":
    # ========== 参数可调区域 ==========
    # 结构元素形状: MORPH_ELLIPSE(椭圆), MORPH_RECT(矩形), MORPH_CROSS(十字)
    SHAPE = cv2.MORPH_RECT
    # 结构元素尺寸: 根据目标大小设定，通常略大于目标直径 (像素)
    KSIZE = 3
    # 局部自适应阈值块大小: 奇数，大致为背景局部区域尺寸
    BLOCK = 11        # 自动调整为奇数，10 → 11
    # 阈值常数: 均值 - C, 增大则抑制更多弱目标，减少则保留更多
    CONST_C = 3
    # 是否保存 Top-Hat 增强图 (用于可视化对比)
    SAVE_ENHANCE = True

    batch_tophat_adaptive(
        input_folder="./images",
        output_folder="./tophat_results",
        morph_shape=SHAPE,
        morph_ksize=KSIZE,
        block_size=BLOCK,
        C=CONST_C,
        save_enhanced=SAVE_ENHANCE
    )