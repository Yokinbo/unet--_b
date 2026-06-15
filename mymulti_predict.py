import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

import archs
from dataset import is_tif_image, preprocess_image, read_image
from multispectral_config import normalization_config, selected_bands, vis_bands


EDITABLE_CONFIG = {
    "name": "6band",
    "input_path": r"论文制图\测试图\原图",
    "label_dir": r"论文制图\测试图\label标签",
    "preview_dir": r"论文制图\测试图\原图rgb",
    "output_dir": r"论文制图\6band测试结果",
    "device": "cuda:0",
    "threshold": 0.5,
    "suffixes": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    "label_suffixes": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "save_mask": True,
    "save_overlay": True,
    "save_prob": False,
    "save_confusion": True,
}


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError("failed to encode image for: %s" % path)
    encoded.tofile(str(path))


def load_config(name):
    config_path = Path("models") / name / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError("config not found: %s" % config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.setdefault("selected_bands", list(selected_bands))
    config.setdefault("normalization_config", normalization_config)
    config.setdefault("vis_bands", list(vis_bands))
    config.setdefault("deep_supervision", False)
    config["input_channels"] = len(config["selected_bands"])
    return config


def build_model(name, config, device):
    if config["arch"] not in archs.__dict__:
        raise ValueError("unknown architecture in config: %s" % config["arch"])

    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config.get("deep_supervision", False),
    )

    checkpoint_path = Path("models") / name / "model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)

    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_input_images(input_path, suffixes):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        suffix_set = {suffix.lower() for suffix in suffixes}
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in suffix_set
        )
    raise FileNotFoundError("input path does not exist: %s" % input_path)


def read_predict_image(image_path, config):
    img_ext = image_path.suffix.lower()
    image = read_image(str(image_path), img_ext, config["selected_bands"])
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] != config["input_channels"]:
        raise ValueError(
            "Read %d channels from %s, but model expects %d."
            % (image.shape[-1], image_path, config["input_channels"])
        )
    return preprocess_image(
        image,
        img_ext,
        config=config["normalization_config"],
        already_reflectance=False,
    )


def stretch_to_uint8(rgb):
    rgb = rgb.astype(np.float32)
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for channel in range(rgb.shape[-1]):
        band = rgb[:, :, channel]
        low = np.percentile(band, 2)
        high = np.percentile(band, 98)
        if high > low:
            out[:, :, channel] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return out


def read_rgb_preview_file(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("failed to read preview image: %s" % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_preview_image(image_path, config, preview_dir=""):
    if preview_dir:
        preview_dir = Path(preview_dir)
        for suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            preview_path = preview_dir / (image_path.stem + suffix)
            if preview_path.exists():
                return read_rgb_preview_file(preview_path)

    img_ext = image_path.suffix.lower()
    if is_tif_image(img_ext):
        preview = read_image(str(image_path), img_ext, config.get("vis_bands", [3, 2, 1]))
        return stretch_to_uint8(preview)

    return read_rgb_preview_file(image_path)


def prepare_tensor(image, config, device):
    target_h = int(config["input_h"])
    target_w = int(config["input_w"])
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(device)


def infer_probability(model, image_tensor, config):
    output = model(image_tensor)
    if isinstance(output, (list, tuple)):
        output = output[-1]

    if int(config["num_classes"]) == 1:
        prob = torch.sigmoid(output)[0, 0]
    else:
        probs = torch.softmax(output, dim=1)
        class_index = 1 if probs.shape[1] > 1 else 0
        prob = probs[0, class_index]
    return prob.detach().cpu().numpy().astype(np.float32)


def resize_to_original(array, shape, interpolation):
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


def save_prediction_outputs(prob, preview_img, threshold, mask_path, overlay_path, prob_path):
    pred_mask = (prob >= threshold).astype(np.uint8)

    if mask_path is not None:
        imwrite_unicode(mask_path, pred_mask * 255)

    if overlay_path is not None:
        overlay = preview_img.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255
        overlay = np.where(pred_mask[..., None] > 0, (0.55 * overlay + 0.45 * red), overlay)
        imwrite_unicode(overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

    if prob_path is not None:
        imwrite_unicode(prob_path, np.clip(prob * 255.0, 0, 255).astype(np.uint8))

    return pred_mask


def find_label_for_image(image_path, label_dir, label_suffixes):
    if not label_dir:
        return None
    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError("label_dir does not exist: %s" % label_dir)

    for suffix in label_suffixes:
        label_path = label_dir / (image_path.stem + suffix)
        if label_path.exists():
            return label_path
    return None


def read_label_mask(label_path, target_shape):
    try:
        label = np.array(Image.open(label_path).convert("L"))
    except Exception as exc:
        raise FileNotFoundError("failed to read label: %s" % label_path) from exc
    if label.shape != target_shape:
        label = cv2.resize(label, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (label > 0).astype(np.uint8)


def save_confusion_outputs(pred_mask, label_mask, label_view_path, confusion_path):
    if label_view_path is not None:
        imwrite_unicode(label_view_path, label_mask.astype(np.uint8) * 255)

    pred01 = (pred_mask > 0).astype(np.uint8)
    label01 = (label_mask > 0).astype(np.uint8)
    rgb = np.zeros((label01.shape[0], label01.shape[1], 3), dtype=np.uint8)

    tp = (pred01 == 1) & (label01 == 1)
    fp = (pred01 == 1) & (label01 == 0)
    fn = (pred01 == 0) & (label01 == 1)

    rgb[tp] = [255, 255, 255]
    rgb[fp] = [0, 0, 255]
    rgb[fn] = [255, 0, 0]
    imwrite_unicode(confusion_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def output_paths(output_dir, image_stem, args):
    output_dir = Path(output_dir)
    return {
        "mask": output_dir / "模型预测mask_0-255" / ("%s_mask.png" % image_stem) if args.save_mask else None,
        "overlay": output_dir / "红色半透明叠加图" / ("%s_overlay.png" % image_stem) if args.save_overlay else None,
        "prob": output_dir / "模型预测概率图" / ("%s_prob.png" % image_stem) if args.save_prob else None,
        "label": output_dir / "人工标签可视化0-255" / ("%s_label.png" % image_stem) if args.save_confusion else None,
        "confusion": output_dir / "TP_FP_FN_TN彩色误差图" / ("%s_confusion.png" % image_stem) if args.save_confusion else None,
    }


def predict_one_image(model, image_path, config, args, device, warmup_done):
    input_image = read_predict_image(image_path, config)
    preview_img = make_preview_image(image_path, config, args.preview_dir)
    original_shape = input_image.shape[:2]

    if preview_img.shape[:2] != original_shape:
        preview_img = cv2.resize(preview_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

    image_tensor = prepare_tensor(input_image, config, device)

    with torch.no_grad():
        if not warmup_done[0]:
            init_img = torch.zeros_like(image_tensor)
            model(init_img)
            warmup_done[0] = True

        t_start = time_synchronized()
        prob = infer_probability(model, image_tensor, config)
        t_end = time_synchronized()

    prob = resize_to_original(prob, original_shape, cv2.INTER_LINEAR)
    paths = output_paths(args.output_dir, image_path.stem, args)
    pred_mask = save_prediction_outputs(
        prob,
        preview_img,
        args.threshold,
        paths["mask"],
        paths["overlay"],
        paths["prob"],
    )

    if paths["confusion"] is not None:
        label_path = find_label_for_image(image_path, args.label_dir, args.label_suffixes)
        if label_path is None:
            print("[warn] no label found for %s, skip confusion map" % image_path.name)
        else:
            label_mask = read_label_mask(label_path, pred_mask.shape)
            save_confusion_outputs(pred_mask, label_mask, paths["label"], paths["confusion"])

    print("[done] %s inference=%.4fs" % (image_path.name, t_end - t_start))


def parse_args():
    parser = argparse.ArgumentParser(description="UNet++ multispectral prediction")
    parser.add_argument("--name", default=EDITABLE_CONFIG["name"], help="model folder under models/")
    parser.add_argument("--input-path", default=EDITABLE_CONFIG["input_path"], help="input image file or folder")
    parser.add_argument("--label-dir", default=EDITABLE_CONFIG["label_dir"], help="manual label folder")
    parser.add_argument("--preview-dir", default=EDITABLE_CONFIG["preview_dir"], help="optional RGB preview folder")
    parser.add_argument("--output-dir", default=EDITABLE_CONFIG["output_dir"], help="output folder")
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"], help="prediction device")
    parser.add_argument("--threshold", default=EDITABLE_CONFIG["threshold"], type=float, help="binary threshold")
    parser.add_argument("--suffixes", nargs="+", default=EDITABLE_CONFIG["suffixes"], help="image suffixes")
    parser.add_argument("--label-suffixes", nargs="+", default=EDITABLE_CONFIG["label_suffixes"], help="label suffixes")
    parser.add_argument("--save-mask", action="store_true", default=EDITABLE_CONFIG["save_mask"])
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask")
    parser.add_argument("--save-overlay", action="store_true", default=EDITABLE_CONFIG["save_overlay"])
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay")
    parser.add_argument("--save-prob", action="store_true", default=EDITABLE_CONFIG["save_prob"])
    parser.add_argument("--no-save-prob", action="store_false", dest="save_prob")
    parser.add_argument("--save-confusion", action="store_true", default=EDITABLE_CONFIG["save_confusion"])
    parser.add_argument("--no-save-confusion", action="store_false", dest="save_confusion")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.name)
    image_paths = collect_input_images(args.input_path, args.suffixes)
    if not image_paths:
        raise FileNotFoundError("No supported images found in: %s" % args.input_path)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.name, config, requested_device)

    print("Current UNet++ multispectral prediction config:")
    print("  model name    : %s" % args.name)
    print("  arch          : %s" % config["arch"])
    print("  input size    : %s x %s" % (config["input_h"], config["input_w"]))
    print("  selected_bands: %s" % config["selected_bands"])
    print("  in_channels   : %s" % config["input_channels"])
    print("  weights       : %s" % (Path("models") / args.name / "model.pth"))
    print("  input_path    : %s" % args.input_path)
    print("  label_dir     : %s" % (args.label_dir or "(disabled)"))
    print("  preview_dir   : %s" % (args.preview_dir or "(disabled)"))
    print("  image_count   : %d" % len(image_paths))
    print("  output_dir    : %s" % args.output_dir)
    print("  device        : %s" % requested_device)

    warmup_done = [False]
    for image_path in image_paths:
        predict_one_image(model, image_path, config, args, requested_device, warmup_done)

    print("Saved outputs to: %s" % args.output_dir)


if __name__ == "__main__":
    main()
