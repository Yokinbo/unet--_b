import argparse
import os
from glob import glob

import cv2
import numpy as np

from dataset import preprocess_image, read_image
from multispectral_config import (
    dataset_name,
    image_ext,
    input_channels,
    mask_ext,
    normalization_config,
    selected_band_names,
    selected_bands,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=dataset_name, help="VOC dataset root, for example VOCdevkit/VOC2007")
    parser.add_argument("--img_ext", default=image_ext, help="image extension, for example .tif or .jpg")
    parser.add_argument("--mask_ext", default=mask_ext, help="mask extension, usually .png")
    parser.add_argument("--image_id", default=None, help="optional image id without extension")
    return parser.parse_args()


def find_image_id(dataset_root, img_ext, image_id=None):
    # 第五步新增：训练前数据读取自检。
    # 如果没有手动指定 image_id，就从 VOC/JPEGImages 下找第一张当前后缀的影像。
    # 这样可以快速确认 tif 能不能被 rasterio 读取，selected_bands 是否越界。
    image_dir = os.path.join(dataset_root, "JPEGImages")
    if image_id is not None:
        return image_id

    image_paths = sorted(glob(os.path.join(image_dir, "*" + img_ext)))
    if not image_paths:
        raise FileNotFoundError(
            "No images found in %s with extension %s" % (image_dir, img_ext)
        )
    return os.path.splitext(os.path.basename(image_paths[0]))[0]


def main():
    args = parse_args()
    image_id = find_image_id(args.dataset, args.img_ext, args.image_id)

    image_path = os.path.join(args.dataset, "JPEGImages", image_id + args.img_ext)
    mask_path = os.path.join(args.dataset, "SegmentationClass", image_id + args.mask_ext)

    # 第五步新增：这里直接复用 dataset.py 的 read_image / preprocess_image。
    # 自检脚本和真实训练走同一套读取、选波段、归一化逻辑，
    # 才能提前发现训练时会遇到的通道数或标准化配置问题。
    image = read_image(image_path, args.img_ext, selected_bands)
    processed = preprocess_image(image, args.img_ext, normalization_config)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError("Mask not found or unreadable: %s" % mask_path)

    print("dataset_root       :", args.dataset)
    print("image_id           :", image_id)
    print("image_path         :", image_path)
    print("mask_path          :", mask_path)
    print("selected_bands     :", selected_bands)
    print("selected_band_names:", selected_band_names)
    print("expected_channels  :", input_channels)
    print("raw_image_shape    :", image.shape)
    print("processed_shape    :", processed.shape)
    print("mask_shape         :", mask.shape)
    print("raw_dtype          :", image.dtype)
    print("processed_dtype    :", processed.dtype)
    print("processed_min/max  :", float(np.min(processed)), float(np.max(processed)))
    print("processed_mean/std :", float(np.mean(processed)), float(np.std(processed)))
    print("mask_unique_values :", np.unique(mask)[:20])

    if processed.shape[-1] != input_channels:
        raise ValueError(
            "Processed channel count is %d, but input_channels is %d."
            % (processed.shape[-1], input_channels)
        )

    print("check_result       : OK")


if __name__ == "__main__":
    main()
