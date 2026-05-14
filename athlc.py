import os
import cv2
import numpy as np
from glob import glob


def athlc_detect(
    image_path,
    output_dir,
    min_ksize=3,
    max_ksize=15,
    kstep=2,
    block_size=11,
    C=2,
    save_enhanced=True
):
    """
    ATHLC (Adaptive Top-Hat based on Local Contrast)
    基于局部对比度的自适应双结构元素 Top-Hat 红外小目标检测
    -----------------------------------------------------------
    1. 局部对比度图计算: 使用多尺度局部标准差估计候选目标尺寸，
       对比度高的区域更有可能包含小目标。
    2. 自适应双结构元素 Top-Hat: 根据局部对比度自适应选择腐蚀核
       和膨胀核的尺寸（腐蚀核 <= 膨胀核），形成双结构元素对，
       比传统单一结构元素对复杂背景具有更强的适应能力。
    3. 局部自适应阈值: 对增强图进行高斯加权局部阈值二值化。
    4. 输出结果包含:
       - 二值检测结果（_binary.png）
       - 归一化的 ATHLC 增强图（_athlc.png，用于观察效果）

    参数说明:
        min_ksize   : 结构元素最小尺寸（奇数）
        max_ksize   : 结构元素最大尺寸（奇数）
        kstep       : 尺度步长
        block_size  : 自适应阈值邻域大小（奇数）
        C           : 阈值常数（均值 - C），值越大检测到的目标越少
        save_enhanced: 是否保存 ATHLC 增强图
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    img = img.astype(np.float32)
    h, w = img.shape

    # ----- 1. 计算局部对比度图（指导自适应核选择） -----
    # 使用局部标准差作为对比度度量，对比度高的区域对应潜在目标
    local_mean = cv2.blur(img, (block_size, block_size))
    local_sqr_mean = cv2.blur(img * img, (block_size, block_size))
    local_std = np.sqrt(np.maximum(local_sqr_mean - local_mean * local_mean, 0))

    # 归一化对比度图到 [min_ksize, max_ksize] 范围，用于指导核尺寸
    contrast_norm = cv2.normalize(local_std, None, 0, 1, cv2.NORM_MINMAX)

    # ----- 2. 多尺度双结构元素 Top-Hat 变换 -----
    # 收集多个尺度下的 Top-Hat 响应，按局部对比度加权融合
    athlc_response = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    # 确保尺寸参数为奇数
    if min_ksize % 2 == 0:
        min_ksize += 1
    if max_ksize % 2 == 0:
        max_ksize += 1

    for k in range(min_ksize, max_ksize + 1, kstep):
        if k % 2 == 0:
            continue

        # 双结构元素：腐蚀核略小于膨胀核，更好地区分目标与背景
        erode_k = max(k - 2, 3)
        dilate_k = k

        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))

        # 开运算 = 先腐蚀后膨胀（用不同的核实现双结构元素效果）
        eroded = cv2.erode(img.astype(np.uint8), kernel_e)
        opened = cv2.dilate(eroded, kernel_d).astype(np.float32)

        # Top-Hat = 原图 - 开运算结果
        tophat_k = np.maximum(img - opened, 0)

        # 计算该尺度下的权重（与局部对比度匹配程度越高的尺度权重越大）
        target_scale = (k - min_ksize) / max(max_ksize - min_ksize, 1)
        scale_diff = np.abs(contrast_norm - target_scale)
        weight_k = np.exp(-scale_diff * scale_diff / 0.1)  # 高斯权重

        athlc_response += tophat_k * weight_k
        weight_sum += weight_k

    # 加权平均融合
    athlc = np.divide(athlc_response, weight_sum + 1e-8)

    # ----- 3. 局部自适应阈值二值化 -----
    athlc_u8 = cv2.normalize(athlc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        athlc_u8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # ----- 4. 保存结果 -----
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    binary_file = os.path.join(output_dir, f"{base_name}_binary.png")
    cv2.imwrite(binary_file, binary)

    if save_enhanced:
        norm = cv2.normalize(athlc, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_file = os.path.join(output_dir, f"{base_name}_athlc.png")
        cv2.imwrite(enhanced_file, norm.astype(np.uint8))

    print(f"已处理: {base_name}")


def batch_athlc(
    input_folder,
    output_folder,
    min_ksize=3,
    max_ksize=5,
    kstep=2,
    block_size=11,
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
        athlc_detect(
            f,
            output_folder,
            min_ksize=min_ksize,
            max_ksize=max_ksize,
            kstep=kstep,
            block_size=block_size,
            C=C,
            save_enhanced=save_enhanced
        )
    print("批量处理完成。")


if __name__ == "__main__":
    # ========== 参数可调区域 ==========
    # 结构元素最小尺寸: 应略小于最小目标直径（奇数）
    MIN_K = 5
    # 结构元素最大尺寸: 应略大于最大目标直径（奇数）
    MAX_K = 7
    # 尺度步长: 尺度间隔，越小越精细但计算量越大
    KSTEP = 2
    # 局部自适应阈值块大小: 奇数，背景局部区域尺寸
    BLOCK = 11
    # 阈值常数: 均值 - C, 增大则抑制更多弱目标
    CONST_C = 10
    # 是否保存 ATHLC 增强图
    SAVE_ENHANCE = True

    batch_athlc(
        input_folder="./images",
        output_folder="./athlc_results",
        min_ksize=MIN_K,
        max_ksize=MAX_K,
        kstep=KSTEP,
        block_size=BLOCK,
        C=CONST_C,
        save_enhanced=SAVE_ENHANCE
    )