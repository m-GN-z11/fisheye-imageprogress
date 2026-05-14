# 红外小目标检测

这个仓库里有7中去畸变后的图像处理方法，对于复杂场景地目标检测效果其实都不够好，也没有对效果做定量比较

---

## 算法总览

| 算法 | 文件 | 核心策略 |
|------|------|----------|
| Top-Hat + 局部自适应阈值 | `tophat.py` | 形态学白顶帽变换 + 自适应阈值 |
| SWLC | `swlc.py` | Top-Hat + 自适应阈值 + 完整目标分析 |
| ATHLC | `athlc.py` | 局部对比度指导的自适应双结构元素 Top-Hat |
| RDLFC | `rdlfc.py` | 三层窗口比值-差分乘积对比度 |
| TLLCM | `tllcm.py` | DoG 滤波 + 改进三层局部对比度 |
| WIDLCM | `widlcm.py` | 加权改进双局部对比度融合 |
| WLRDC | `wlrdc.py` | 八方向邻域比值-差分对比度 + 背景加权 |

---

## 1. Top-Hat + 局部自适应阈值 (`tophat.py`)

### 算法原理
经典的红外小目标检测流程，通过形态学变换增强目标区域，再利用局部自适应阈值完成分割。

### 处理流程

1. **白顶帽变换 (White Top-Hat)**
   - 使用椭圆形/矩形/十字结构元素对原图进行开运算（先腐蚀后膨胀），估计局部背景
   - 原图减去开运算结果：`tophat = 原图 - open(原图)`
   - 差分后亮目标被突出，大面积背景被抑制

2. **局部自适应阈值**
   - 对 Top-Hat 增强图进行高斯加权局部阈值二值化
   - 阈值公式：`T = mean(local) - C`
   - 局部阈值分割能有效应对画面光照不均

3. **目标检测、合并与标记**
   - 对二值图提取外部轮廓，通过面积过滤去除噪声
   - 使用**中心距离合并**（基于并查集），将中心距离 ≤ 阈值的邻近矩形合并为同一目标
   - 为每个目标生成连续编号，绘制绿色矩形框与编号标签

4. **输出结果**
   - `{name}_binary.png`：二值检测结果
   - `{name}_tophat.png`：归一化的 Top-Hat 增强图
   - `{name}_marked.png`：带编号标记的检测结果图
   - `{name}_targets.csv`：目标属性表（面积、中心坐标、边界框、平均/最大强度、Top-Hat 均值）

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `morph_shape` | 结构元素形状（椭圆/矩形/十字） | `MORPH_RECT` |
| `morph_ksize` | 结构元素尺寸，略大于目标 | 3~7 |
| `block_size` | 自适应阈值邻域大小（奇数） | 11~15 |
| `C` | 阈值偏移量，越大目标越少 | 2~3 |
| `merge_distance` | 中心距离合并阈值（像素） | 10 |
| `min_area` | 最小连通域面积 | 2 |

---

## 2. SWLC (`swlc.py`)

### 算法原理
SWLC（Sliding Window Local Contrast）基于 Top-Hat 变换与局部自适应阈值的组合框架，在经典流程基础上完善了目标分析模块，实现对检测结果的完整量化与标注。

### 处理流程

1. **形态学 Top-Hat 变换**
   - 同 `tophat.py`，通过结构元素估计背景并差分增强

2. **局部自适应阈值**
   - 对 Top-Hat 增强图进行二值化，自动处理光照不均

3. **目标分析与输出**
   - 轮廓提取 → 面积过滤 → 中心距离合并 → 属性计算
   - 为每个检测目标提取：边界框、面积、中心坐标、原图平均/最大强度、Top-Hat 均值
   - 生成带编号的标记图像和结构化 CSV 数据

### 与 tophat.py 的区别
- 采用更简洁的矩形 ROI 提取方式（`img[y:y+h, x:x+w]`）
- 分析模块架构更清晰，便于扩展
- 批量处理函数支持 `**kwargs` 透传，参数配置更灵活

### 关键参数
与 `tophat.py` 相同，额外强调：
- `analyze=True`：开启目标分析
- `save_marked=True`：保存带编号标记图像
- `save_enhanced=True`：保存增强图

---

## 3. ATHLC - 基于局部对比度的自适应双结构元素 Top-Hat (`athlc.py`)

### 算法原理
ATHLC（Adaptive Top-Hat based on Local Contrast）通过局部对比度估计目标尺度，自适应选择腐蚀核与膨胀核尺寸，实现比传统单一结构元素更强的复杂背景适应能力。

### 处理流程

1. **局部对比度图计算**
   - 使用局部标准差作为对比度度量：`std = sqrt(E[x^2] - E[x]^2)`
   - 对比度高的区域更有可能包含小目标
   - 将对比度图归一化到 `[min_ksize, max_ksize]` 范围，指导核尺寸选择

2. **多尺度双结构元素 Top-Hat 变换**
   - 在 `[min_ksize, max_ksize]` 范围内遍历多个尺度
   - 对每个尺度构造**双结构元素对**：腐蚀核略小于膨胀核（`erode_k = k - 2`, `dilate_k = k`）
   - 计算 Top-Hat 响应：`tophat_k = max(原图 - opened, 0)`
   - 根据局部对比度与尺度匹配程度计算高斯权重
   - 加权融合所有尺度响应：`athlc = sum(tophat_k * weight_k) / sum(weight_k)`

3. **局部自适应阈值二值化**
   - 对融合后的 ATHLC 增强图进行高斯加权局部阈值分割

4. **输出结果**
   - `{name}_binary.png`：二值检测结果
   - `{name}_athlc.png`：归一化的 ATHLC 增强图

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `min_ksize` | 结构元素最小尺寸（奇数） | 3~5 |
| `max_ksize` | 结构元素最大尺寸（奇数） | 7~15 |
| `kstep` | 尺度步长 | 2 |
| `block_size` | 自适应阈值邻域大小 | 11 |
| `C` | 阈值常数 | 2~10 |

---

## 4. RDLFC - 比值-差分局部特征对比度 (`rdlfc.py`)

### 算法原理
RDLFC（Ratio-Difference Local Feature Contrast）采用三层嵌套窗口结构，通过比值与差分的乘积构建显著性度量，强调目标与背景在强度和比例上的双重差异。

### 处理流程

1. **三层窗口均值计算**
   - 内窗（目标区域）：`target_size x target_size`
   - 中窗（保护环）：`mid_size x mid_size`
   - 外窗（背景环）：`outer_size x outer_size`
   - 使用 `cv2.blur` 计算各层均值

2. **环域均值计算**
   - 中间环均值（排除内窗）：`mid_ring = (mid_mean * mid_area - inner_mean * inner_area) / (mid_area - inner_area)`
   - 外环均值（排除中窗）：`outer_ring = (outer_mean * outer_area - mid_mean * mid_area) / (outer_area - mid_area)`

3. **比值-差分乘积对比度**
   - 内-中环对比度：`C1 = max(0, (inner - mid_ring) * (inner / mid_ring))`
   - 内-外环对比度：`C2 = max(0, (inner - outer_ring) * (inner / outer_ring))`
   - 显著图：`saliency = C1 * C2`（两者同时显著时响应最强）

4. **阈值分割与面积滤波**
   - 全局阈值：`thresh = max(saliency) * thresh_ratio`
   - 连通域分析过滤小面积噪声

5. **目标分析与输出**
   - 对滤波后的连通域重新计算精确属性
   - 输出标记图和 CSV（含显著度均值 `saliency_mean`）

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `target_size` | 目标区域内窗边长（奇数） | 3 |
| `mid_size` | 中间保护环边长（奇数） | 7 |
| `outer_size` | 背景外环边长（奇数） | 11 |
| `thresh_ratio` | 显著图阈值比例 | 0.4~0.5 |
| `min_area` | 最小连通域面积 | 2~3 |

---

## 5. TLLCM - 改进的三层局部对比度 (`tllcm.py`)

### 算法原理
TLLCM（Three-Layer Local Contrast Method）在三层窗口对比度基础上引入 DoG（Difference of Gaussians）预处理，采用 `min(T-M, T-O)` 的保守显著性度量，并配合形态学开运算去除虚警。

### 处理流程

1. **DoG 滤波预处理**
   - `DoG = Gaussian(sigma1) - Gaussian(sigma2)`
   - 抑制高频噪声，增强 blob-like 目标

2. **三层均值计算（在 DoG 图上）**
   - 内窗均值 `inner_mean`：目标区域
   - 中窗均值 `mid_mean`：含保护环
   - 外窗均值 `outer_mean`：含背景

3. **环域均值**
   - 计算中间环和外环的均值（排除内层区域）

4. **保守对比度度量**
   - `diff1 = inner_mean - mid_ring`
   - `diff2 = inner_mean - outer_ring`
   - `saliency = min(diff1, diff2)`
   - 取最小值可抑制仅在某一层显著的虚假目标

5. **形态学后处理**
   - 阈值二值化后，使用 3x3 椭圆核进行开运算去噪

6. **目标分析**
   - 轮廓提取 → 面积过滤 → 中心距离合并
   - 输出标记图和 CSV（含显著度均值）

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `inner_size` | 内窗边长 | 3~5 |
| `mid_size` | 中窗边长 | 7 |
| `outer_size` | 外窗边长 | 11 |
| `dog_sigma1` | DoG 小尺度高斯标准差 | 1.0 |
| `dog_sigma2` | DoG 大尺度高斯标准差 | 4.0~8.0 |
| `thresh_ratio` | 阈值比例 | 0.5 |
| `morph_open` | 是否进行开运算去噪 | True |

---

## 6. WIDLCM - 加权改进双局部对比度 (`widlcm.py`)

### 算法原理
WIDLCM（Weighted Improved Dual Local Contrast Method）在三层窗口框架下计算两个独立的对比度分量，通过可调的加权系数融合，灵活平衡近邻背景与远邻背景对显著性的贡献。

### 处理流程

1. **三层窗口均值计算**
   - 同 RDLFC，计算内窗、中窗、外窗的均值

2. **环域均值**
   - 中间环均值 `mid_ring` 和外环均值 `outer_ring`

3. **双对比度分量**
   - `C1 = (inner - mid_ring) * (inner / mid_ring)` —— 目标与近邻背景的比值-差分对比度
   - `C2 = (inner - outer_ring) * (inner / outer_ring)` —— 目标与远邻背景的比值-差分对比度

4. **加权融合**
   - `saliency = weight_inner * C1 + (1 - weight_inner) * C2`
   - `weight_inner` 控制近邻背景的权重（默认 0.6）

5. **面积滤波与目标分析**
   - 连通域面积过滤
   - 中心距离合并、标记、CSV 导出

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `inner_size` | 内窗边长 | 3 |
| `mid_size` | 中窗边长 | 7 |
| `outer_size` | 外窗边长 | 11 |
| `weight_inner` | C1 的权重（0~1） | 0.6 |
| `thresh_ratio` | 阈值比例 | 0.5 |
| `min_area` | 最小连通域面积 | 2 |

---

## 7. WLRDC - 加权局部比值-差分对比度 (`wlrdc.py`)

### 算法原理
WLRDC（Weighted Local Ratio-Difference Contrast）采用八方向邻域对比度策略，以目标区域为中心，向 8 个方向偏移提取邻域背景，取最小对比度作为显著性度量，并用背景复杂度进行自适应加权。

### 处理流程

1. **中心区域均值计算**
   - 使用 `target_size x target_size` 的平均滤波核计算中心均值 `center_mean`

2. **八方向邻域对比度**
   - 对 8 个方向（上下左右 + 4 对角线）分别以 `pad_size` 为偏移量提取邻域
   - 对每个方向计算邻域均值 `neighbor_mean`
   - 对比度公式：`contrast = max(0, (center - neighbor) * (center / neighbor))`
   - 显著图取八方向最小值：`saliency = min(contrast_1, ..., contrast_8)`
   - 最小值策略确保只有所有方向都显著的区域才被标记

3. **背景复杂度加权**
   - 计算局部背景标准差 `bg_std`
   - 权重：`weight = 1 / (1 + bg_std / weight_sigma)`
   - 背景复杂区域（`bg_std` 大）权重降低，抑制边缘和纹理区域
   - 最终显著图：`saliency = saliency * weight`

4. **阈值分割与面积滤波**
   - 全局阈值 + 连通域面积过滤

5. **目标分析**
   - 轮廓提取 → 中心距离合并 → 标记与 CSV 导出

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `target_size` | 中心目标区域边长 | 7 |
| `pad_size` | 邻域偏移距离 | 5 |
| `weight_sigma` | 背景加权系数 | 10.0 |
| `thresh_ratio` | 阈值比例 | 0.5 |
| `min_area` | 最小连通域面积 | 2 |

---

## 公共模块说明

### 中心距离矩形合并 (`merge_rects_by_center_distance`)

多个算法（Top-Hat、SWLC、TLLCM、WIDLCM、WLRDC）共享此模块，用于将邻近的检测框合并为同一目标。

**原理：**
- 计算所有边界框的中心坐标
- 使用**并查集（Union-Find）**数据结构，将中心欧氏距离 ≤ `distance_threshold` 的矩形归为同一组
- 每组生成能覆盖组内所有矩形的最小外接矩形

**适用场景：**
- 单个目标被分割成多个邻近连通域时
- 避免对同一目标重复计数

---

## 通用参数说明

以下参数在多种算法中通用：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `analyze` | 是否开启目标分析（轮廓、属性、CSV） | `True` |
| `save_marked` | 是否保存带编号标记图像 | `True` |
| `save_enhanced` | 是否保存增强/显著图 | `True` |
| `min_area` | 最小连通域面积阈值 | 2 |
| `merge_distance` | 中心距离合并阈值 | 10 |

---

## 依赖

- **OpenCV** (`cv2`)：图像读取、形态学操作、滤波、轮廓提取、连通域分析
- **NumPy**：数值计算、数组操作
- **Python 标准库**：`os`、`csv`、`glob`

安装：
```bash
pip install opencv-python numpy
```
