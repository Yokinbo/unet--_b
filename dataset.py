import os
import random

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


def preprocess_image(image, img_ext, config=None, already_reflectance=False):
    # 第二步修改：统一图像预处理。
    # 普通 RGB 图像仍沿用旧逻辑 /255。
    # Sentinel-2 多光谱 tif 通常是 DN = reflectance * 10000，
    # 因此先除以 reflectance_scale，再按配置可选 clip 和 mean/std。
    image = image.astype("float32")

    if not is_tif_image(img_ext):
        return image / 255.0

    config = config or normalization_config
    reflectance_scale = float(config.get("reflectance_scale", 10000.0))
    if reflectance_scale > 0 and not already_reflectance:
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


def use_training_augmentation(image, img_ext, augmentation_config):
    return (
        augmentation_config.get("enabled", False)
        and is_tif_image(img_ext)
        and isinstance(image, np.ndarray)
        and image.ndim == 3
    )


def image_to_reflectance(image, normalization_cfg):
    reflectance = image.astype(np.float32, copy=False)
    scale = float(normalization_cfg.get("reflectance_scale", 10000.0))
    if scale > 0:
        reflectance = reflectance / scale
    return reflectance


def augment_training_sample(image, mask, normalization_cfg, augmentation_config):
    reflectance = image_to_reflectance(image, normalization_cfg)

    if random.random() < augmentation_config.get("geometry_prob", 0.0):
        reflectance, mask = augment_geometry(reflectance, mask)

    if random.random() < augmentation_config.get("scale_prob", 0.0):
        reflectance, mask = augment_random_scale(reflectance, mask, augmentation_config)

    if random.random() < augmentation_config.get("reflectance_prob", 0.0):
        reflectance = augment_reflectance(reflectance, augmentation_config)

    if random.random() < augmentation_config.get("shadow_prob", 0.0):
        reflectance = augment_shadow(reflectance, augmentation_config)

    if random.random() < augmentation_config.get("noise_prob", 0.0):
        reflectance = augment_noise(reflectance, augmentation_config)

    return reflectance.astype(np.float32), mask


def augment_geometry(image, mask):
    op = random.choice(["hflip", "vflip", "rot90", "rot180", "rot270"])
    if op == "hflip":
        return np.ascontiguousarray(image[:, ::-1, :]), np.ascontiguousarray(mask[:, ::-1, ...])
    if op == "vflip":
        return np.ascontiguousarray(image[::-1, :, :]), np.ascontiguousarray(mask[::-1, :, ...])
    k = {"rot90": 1, "rot180": 2, "rot270": 3}[op]
    return np.rot90(image, k=k).copy(), np.rot90(mask, k=k).copy()


def augment_reflectance(image, augmentation_config):
    global_low, global_high = augmentation_config.get("reflectance_global_range", [0.90, 1.10])
    band_low, band_high = augmentation_config.get("reflectance_band_range", [0.95, 1.05])
    global_factor = random.uniform(global_low, global_high)
    band_factors = np.random.uniform(
        band_low,
        band_high,
        size=(1, 1, image.shape[2]),
    ).astype(np.float32)
    return image * global_factor * band_factors


def augment_shadow(image, augmentation_config):
    factor_low, factor_high = augmentation_config.get("shadow_factor_range", [0.75, 0.90])
    radius_low, radius_high = augmentation_config.get("shadow_radius_range", [0.25, 0.45])
    h, w = image.shape[:2]
    center_y = random.uniform(-0.5, 0.5)
    center_x = random.uniform(-0.5, 0.5)
    radius = random.uniform(radius_low, radius_high)
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    shadow = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(radius, 1e-6))
    factor = random.uniform(factor_low, factor_high)
    shadow_map = 1.0 - (1.0 - factor) * shadow
    return image * shadow_map[:, :, None]


def augment_noise(image, augmentation_config):
    sigma_low, sigma_high = augmentation_config.get("noise_sigma_range", [0.003, 0.008])
    sigma = random.uniform(sigma_low, sigma_high)
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return image + noise


def augment_random_scale(image, mask, augmentation_config):
    crop_low, crop_high = augmentation_config.get("scale_crop_range", [0.85, 1.00])
    ratio = random.uniform(crop_low, crop_high)
    h, w = image.shape[:2]
    crop_h = max(8, int(h * ratio))
    crop_w = max(8, int(w * ratio))
    top = random.randint(0, max(0, h - crop_h))
    left = random.randint(0, max(0, w - crop_w))

    image_crop = image[top:top + crop_h, left:left + crop_w, :]
    mask_crop = mask[top:top + crop_h, left:left + crop_w, ...]
    image = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    if image.ndim == 2:
        image = image[:, :, None]
    if mask.ndim == 2:
        mask = mask[:, :, None]
    return image.astype(np.float32), mask


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
            normalization_config=normalization_config,
            augmentation_config=None):
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
        self.augmentation_config = augmentation_config or {"enabled": False}

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

        already_reflectance = False
        if use_training_augmentation(img, self.img_ext, self.augmentation_config):
            img, mask = augment_training_sample(
                img,
                mask,
                self.normalization_config,
                self.augmentation_config,
            )
            already_reflectance = True

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = preprocess_image(img, self.img_ext, self.normalization_config, already_reflectance)
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
            normalization_config=normalization_config,
            augmentation_config=None):
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
        self.augmentation_config = augmentation_config or {"enabled": False}

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

        already_reflectance = False
        if use_training_augmentation(img, self.img_ext, self.augmentation_config):
            img, mask = augment_training_sample(
                img,
                mask,
                self.normalization_config,
                self.augmentation_config,
            )
            already_reflectance = True

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = preprocess_image(img, self.img_ext, self.normalization_config, already_reflectance)
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        if mask.max() > 1:
            mask /= 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
