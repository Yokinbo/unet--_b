# UNet++ 多光谱训练说明

这个分支把原来的普通 RGB/JPG 训练流程改成了可支持 Sentinel-2 多光谱 `.tif` 的训练与验证流程。当前默认配置是 6 波段：

```text
B2, B3, B4, B8, B11, B12
```

## 1. 主要改动

### 第一步：统一配置入口

新增：

```text
multispectral_config.py
```

这里集中管理：

```python
band_mode
selected_bands
selected_band_names
input_channels
vis_bands
normalization_config
```

切换实验时优先改这个文件，例如：

```python
band_mode = "rgb"
band_mode = "4band"
band_mode = "6band"
```

### 第二步：数据读取支持 tif

修改：

```text
dataset.py
```

原来固定使用 `cv2.imread()` 读取普通 RGB/JPG。现在根据后缀判断：

```text
.jpg/.png 等普通图像 -> cv2.imread()
.tif/.tiff 多光谱影像 -> rasterio.open()
```

多光谱读取时会按 `selected_bands` 选择波段，并把 rasterio 的 `(C, H, W)` 转成训练需要的 `(H, W, C)`。

### 第三步：多光谱预处理

普通 RGB 图像仍使用：

```text
image / 255
```

Sentinel-2 多光谱 `.tif` 使用：

```text
DN / 10000 -> clip -> mean/std
```

对应配置在 `multispectral_config.py` 的 `normalization_configs` 中。论文实验中，`clip_min/clip_max/mean/std` 应只由训练集统计得到，不能使用验证集或测试集，避免数据泄漏。

### 第四步：训练入口接入配置

修改：

```text
train.py
```

训练入口现在会从 `multispectral_config.py` 读取：

```python
dataset_name
image_ext
mask_ext
band_mode
selected_bands
input_channels
normalization_config
```

训练时保存的：

```text
models/<name>/config.yml
```

也会记录这些多光谱实验配置，方便复现实验。

### 第五步：验证入口接入配置

修改：

```text
val.py
```

验证阶段会读取训练保存的 `config.yml`，保持和训练相同的：

```text
影像后缀
输入通道
波段选择
标准化配置
```

同时，验证示例图会尽量按 `B4/B3/B2` 生成真彩色预览。

## 2. 数据集目录结构

当前默认使用 VOC 格式：

```text
VOCdevkit/VOC2007
├── JPEGImages
│   ├── xxx.tif
│   ├── yyy.tif
│   └── ...
├── SegmentationClass
│   ├── xxx.png
│   ├── yyy.png
│   └── ...
└── ImageSets
    └── Segmentation
        ├── train.txt
        └── val.txt
```

注意：虽然目录名仍叫 `JPEGImages`，但多光谱模式下里面放的是 `.tif`。

## 3. 波段约定

当前假设 6 波段 tif 的原始顺序是：

```text
[1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
```

配置文件中的三种模式：

```python
"rgb"   -> [3, 2, 1]        # B4, B3, B2
"4band" -> [1, 2, 3, 4]     # B2, B3, B4, B8
"6band" -> [1, 2, 3, 4, 5, 6]
```

如果你的 tif 波段顺序不同，必须先改 `multispectral_config.py` 中的 `band_options` 和 `band_names`。

## 4. 训练前自检

新增：

```text
check_multispectral_dataset.py
```

运行：

```bash
python check_multispectral_dataset.py
```

它会读取一张样本，打印：

```text
selected_bands
raw_image_shape
processed_shape
mask_shape
processed_min/max
processed_mean/std
mask_unique_values
```

也可以指定某张影像：

```bash
python check_multispectral_dataset.py --image_id xxx
```

如果这里报错，优先检查：

```text
1. rasterio 是否安装
2. tif 后缀是否和 image_ext 一致
3. selected_bands 是否超过 tif 实际波段数
4. mask 是否存在于 SegmentationClass
```

## 5. 训练与验证

训练：

```bash
python train.py
```

验证：

```bash
python val.py --name voc_run2 --split val
```

测试：

```bash
python val.py --name voc_run2 --split test
```

如果你修改了 `TRAIN_DEFAULTS['name']`，验证时也要对应修改 `--name`。

如果不想每次输入命令行参数，可以直接修改 `val.py` 顶部：

```python
VAL_DEFAULTS = {
    'name': 'voc_run2',
    'split': 'val',
}
```

然后直接运行：

```bash
python val.py
```

`--split val` 会读取：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
```

并输出到：

```text
outputs/<name>
```

`--split test` 会读取：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt
```

并输出到：

```text
outputs/<name>_test
```

这样验证集和测试集的指标、预测图不会互相覆盖。

## 6. 依赖

多光谱 tif 读取需要：

```text
rasterio
```

当前依赖写在：

```text
requirements.txt
```

如果训练环境缺依赖，可以在对应 Python 环境中安装：

```bash
pip install -r requirements.txt
```

## 7. 论文实验建议

建议至少做三组消融实验：

```text
RGB:   B4, B3, B2
4band: B2, B3, B4, B8
6band: B2, B3, B4, B8, B11, B12
```

每组实验应保存对应的：

```text
config.yml
log.csv
metrics.yml
metrics.csv
model.pth
```

这样论文中可以清楚说明不同波段组合对光伏提取效果的影响。
