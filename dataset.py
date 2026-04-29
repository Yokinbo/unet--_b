import os

import cv2
import numpy as np
import torch
import torch.utils.data

from multispectral_config import normalization_config, selected_bands


def is_tif_image(img_ext):
    return img_ext.lower() in [".tif", ".tiff"]


def read_image(image_path, img_ext, bands=None):
    # 第一步修改：支持 tif 多光谱读取。
    # 原来这里固定使用 cv2.imread 读取普通 RGB/JPG 图像。
    # 现在根据后缀判断：
    # - tif/tiff 使用 rasterio 读取多波段影像，并支持 selected_bands 选波段
    # - jpg/png/bmp 等普通图像继续使用 cv2.imread，兼容旧 RGB 流程
    if is_tif_image(img_ext):
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "Reading multispectral tif images requires rasterio. "
                "Please install rasterio before training with image_ext='.tif'."
            ) from exc

        with rasterio.open(image_path) as src:
            band_indexes = bands if bands is not None else list(range(1, src.count + 1))
            image = src.read(indexes=band_indexes)  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # -> (H, W, C)
        return image

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or unreadable: %s" % image_path)
    return image


def preprocess_image(image, img_ext, config=None):
    # 第二步修改：统一图像预处理。
    # 普通 RGB 图像仍沿用旧逻辑 /255。
    # Sentinel-2 多光谱 tif 通常是 DN = reflectance * 10000，
    # 因此先除以 reflectance_scale，再按配置可选 clip 和 mean/std。
    image = image.astype("float32")

    if not is_tif_image(img_ext):
        return image / 255.0

    config = config or normalization_config
    reflectance_scale = float(config.get("reflectance_scale", 10000.0))
    if reflectance_scale > 0:
        image = image / reflectance_scale

    if config.get("enable_clip", False):
        clip_min = np.asarray(config["clip_min"], dtype=np.float32).reshape(1, 1, -1)
        clip_max = np.asarray(config["clip_max"], dtype=np.float32).reshape(1, 1, -1)
        if clip_min.shape[-1] != image.shape[-1] or clip_max.shape[-1] != image.shape[-1]:
            raise ValueError(
                "clip_min/clip_max channel count does not match image channels: "
                "%d vs %d" % (clip_min.shape[-1], image.shape[-1])
            )
        image = np.clip(image, clip_min, clip_max)

    if config.get("enable_mean_std", False):
        mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(config["std"], dtype=np.float32).reshape(1, 1, -1)
        if mean.shape[-1] != image.shape[-1] or std.shape[-1] != image.shape[-1]:
            raise ValueError(
                "mean/std channel count does not match image channels: "
                "%d vs %d" % (mean.shape[-1], image.shape[-1])
            )
        image = (image - mean) / np.maximum(std, 1e-6)

    return image


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_ids,
            img_dir,
            mask_dir,
            img_ext,
            mask_ext,
            num_classes,
            transform=None,
            selected_bands=selected_bands,
            normalization_config=normalization_config):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        # 第二步补充：把多光谱配置挂到数据集对象上。
        # 这样 Dataset 和 VOCDataset 都可以使用同一套 selected_bands
        # 与 normalization_config，避免训练/验证读取逻辑不一致。
        self.selected_bands = selected_bands
        self.normalization_config = normalization_config

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = read_image(
            os.path.join(self.img_dir, img_id + self.img_ext),
            self.img_ext,
            self.selected_bands,
        )

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise FileNotFoundError("Mask not found or unreadable: %s" % mask_path)
            mask.append(mask_image[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = preprocess_image(img, self.img_ext, self.normalization_config)
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_ids,
            img_dir,
            mask_dir,
            img_ext='.jpg',
            mask_ext='.png',
            transform=None,
            selected_bands=selected_bands,
            normalization_config=normalization_config):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        # 第二步补充：VOC 格式数据集也接入同一套多光谱配置。
        # 后续 train.py / val.py 只要传入 img_ext='.tif'，
        # 这里就会自动读取 selected_bands 并执行多光谱预处理。
        self.selected_bands = selected_bands
        self.normalization_config = normalization_config

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = read_image(
            os.path.join(self.img_dir, img_id + self.img_ext),
            self.img_ext,
            self.selected_bands,
        )
        mask = cv2.imread(
            os.path.join(self.mask_dir, img_id + self.mask_ext),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(
                "Mask not found or unreadable: %s"
                % os.path.join(self.mask_dir, img_id + self.mask_ext)
            )

        # SegmentationClass stores binary labels as 0/1 in this repo.
        mask = mask[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = preprocess_image(img, self.img_ext, self.normalization_config)
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        if mask.max() > 1:
            mask /= 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
