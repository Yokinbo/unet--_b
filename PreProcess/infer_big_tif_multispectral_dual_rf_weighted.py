"""
Large-area Sentinel-2 photovoltaic mapping with seamless weighted sliding-window inference.

This script is written for the current UNet++_b training pipeline:
- loads models/<name>/config.yml and models/<name>/model.pth
- reads Sentinel-2 bands from the saved training config
- reuses dataset.preprocess_image so inference normalization matches training
- blends overlapping tiles in probability space with a Hann window
- optionally fuses two receptive fields by using local and larger-context windows

Typical usage:
python PreProcess/infer_big_tif_multispectral_dual_rf_weighted.py ^
  --name rgb ^
  --input_tif E:/data/sentinel2_6band.tif ^
  --out_class E:/data/pv_class.tif ^
  --out_conf E:/data/pv_conf.tif ^
  --context_scales 1.0,1.5
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import archs
from dataset import preprocess_image
from multispectral_config import normalization_config, selected_bands, vis_bands


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seamless multispectral big GeoTIFF inference for PV mapping."
    )
    parser.add_argument("--name", default="rgb", help="model folder under models/")
    parser.add_argument("--input_tif", required=True, help="large multispectral GeoTIFF")
    parser.add_argument("--out_class", required=True, help="output uint8 class GeoTIFF")
    parser.add_argument("--out_conf", default="", help="optional output float32 confidence GeoTIFF")
    parser.add_argument("--tile_size", type=int, default=0, help="default: config input_h")
    parser.add_argument("--overlap", type=int, default=64, help="overlap pixels for base tiles")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--context_scales",
        default="1.0,1.5",
        help="comma-separated receptive-field scales, e.g. 1.0 or 1.0,1.5",
    )
    parser.add_argument(
        "--context_weights",
        default="0.65,0.35",
        help="comma-separated weights for context_scales",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="binary PV threshold")
    parser.add_argument("--skip_zero_tiles", action="store_true", default=True)
    parser.add_argument("--no_skip_zero_tiles", action="store_false", dest="skip_zero_tiles")
    parser.add_argument("--skip_zero_ratio", type=float, default=0.98)
    parser.add_argument("--temp_dir", default="tmp_big_tif_infer")
    parser.add_argument("--write_block", type=int, default=2048)
    parser.add_argument("--keep_temp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_config(name):
    config_path = ROOT / "models" / name / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError("Config not found: %s" % config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.setdefault("selected_bands", list(selected_bands))
    config.setdefault("normalization_config", normalization_config)
    config.setdefault("vis_bands", list(vis_bands))
    config["input_channels"] = len(config["selected_bands"])
    return config


def build_model(name, config, device):
    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config.get("deep_supervision", False),
    )
    checkpoint_path = ROOT / "models" / name / "model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_weight_window(tile_size):
    one_d = np.hanning(tile_size).astype(np.float32)
    one_d = np.maximum(one_d, 1e-3)
    return np.outer(one_d, one_d).astype(np.float32)


def should_skip_tile(patch_chw, zero_ratio):
    patch_hwc = np.transpose(patch_chw, (1, 2, 0))
    return float(np.mean(np.all(patch_hwc == 0, axis=2))) >= zero_ratio


def read_context_patch(src, x0, y0, tile_size, scale, band_indexes):
    if scale <= 1.0:
        win = Window(x0, y0, tile_size, tile_size)
        patch = src.read(indexes=band_indexes, window=win, boundless=True, fill_value=0)
        return patch, tile_size, 0

    context_size = int(round(tile_size * scale))
    if context_size % 2 != tile_size % 2:
        context_size += 1
    margin = (context_size - tile_size) // 2
    win = Window(x0 - margin, y0 - margin, context_size, context_size)
    patch = src.read(indexes=band_indexes, window=win, boundless=True, fill_value=0)
    return patch, context_size, margin


def prepare_patch(patch_chw, config, model_size):
    patch_hwc = np.transpose(patch_chw, (1, 2, 0))
    patch_hwc = preprocess_image(
        patch_hwc,
        config["img_ext"],
        config=config["normalization_config"],
        already_reflectance=False,
    )
    if patch_hwc.shape[0] != model_size or patch_hwc.shape[1] != model_size:
        patch_hwc = cv2.resize(patch_hwc, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
        if patch_hwc.ndim == 2:
            patch_hwc = patch_hwc[:, :, None]
    patch_chw = np.transpose(patch_hwc, (2, 0, 1)).astype(np.float32, copy=False)
    return patch_chw


def infer_batch(model, batch_tiles, config, device):
    batch = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(device)
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, (list, tuple)):
            output = output[-1]

        if config["num_classes"] == 1:
            probs = torch.sigmoid(output)
        else:
            probs = torch.softmax(output, dim=1)
    return probs.detach().cpu().numpy().astype(np.float32)


def resize_prob_to_context(prob, context_size):
    if prob.shape[-2:] == (context_size, context_size):
        return prob

    channels = []
    for c in range(prob.shape[0]):
        channels.append(cv2.resize(prob[c], (context_size, context_size), interpolation=cv2.INTER_LINEAR))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def flush_pending(model, pending, score_sum, weight_sum, weight_window, config, device):
    if not pending:
        return 0

    batch_tiles = [item["tensor"] for item in pending]
    probs = infer_batch(model, batch_tiles, config, device)

    for item, prob in zip(pending, probs):
        x0 = item["x0"]
        y0 = item["y0"]
        h = item["h"]
        w = item["w"]
        scale_weight = item["scale_weight"]

        prob = resize_prob_to_context(prob, item["context_size"])
        margin = item["context_margin"]
        prob = prob[:, margin:margin + item["tile_size"], margin:margin + item["tile_size"]]
        prob = prob[:, :h, :w]
        weight = weight_window[:h, :w] * scale_weight

        score_sum[:, y0:y0 + h, x0:x0 + w] += prob * weight[np.newaxis, :, :]
        weight_sum[y0:y0 + h, x0:x0 + w] += weight

    count = len(pending)
    pending.clear()
    return count


def write_outputs(src, args, config, score_sum, weight_sum, channels, height, width):
    profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", compress="lzw", tiled=True, nodata=0)

    ensure_parent(args.out_class)
    if args.out_conf:
        ensure_parent(args.out_conf)

    conf_dst = None
    if args.out_conf:
        conf_profile = src.profile.copy()
        conf_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            compress="lzw",
            tiled=True,
            nodata=0,
        )
        conf_dst = rasterio.open(args.out_conf, "w", **conf_profile)

    try:
        with rasterio.open(args.out_class, "w", **profile) as class_dst:
            block = int(args.write_block)
            for y0 in tqdm(range(0, height, block), desc="writing", ncols=100):
                h = min(block, height - y0)
                weight = np.maximum(weight_sum[y0:y0 + h, :], 1e-6)
                probs = score_sum[:, y0:y0 + h, :] / weight[np.newaxis, :, :]

                if config["num_classes"] == 1:
                    conf = probs[0]
                    pred = (conf >= args.threshold).astype(np.uint8)
                else:
                    pred = np.argmax(probs, axis=0).astype(np.uint8)
                    if channels == 2:
                        conf = probs[1]
                    else:
                        conf = np.max(probs, axis=0)

                out_win = Window(0, y0, width, h)
                class_dst.write(pred[np.newaxis, :, :], window=out_win)
                if conf_dst is not None:
                    conf_dst.write(conf.astype(np.float32, copy=False)[np.newaxis, :, :], window=out_win)
    finally:
        if conf_dst is not None:
            conf_dst.close()


def main():
    args = parse_args()
    scales = parse_float_list(args.context_scales)
    scale_weights = parse_float_list(args.context_weights)
    if not scales:
        raise ValueError("context_scales is empty")
    if len(scale_weights) != len(scales):
        if len(scale_weights) == 1:
            scale_weights = scale_weights * len(scales)
        else:
            raise ValueError("context_weights must match context_scales")
    scale_weights = np.asarray(scale_weights, dtype=np.float32)
    scale_weights = (scale_weights / np.maximum(scale_weights.sum(), 1e-6)).tolist()

    config = load_config(args.name)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(args.name, config, device)

    tile_size = int(args.tile_size or config["input_h"])
    model_size = int(config["input_h"])
    if int(config["input_h"]) != int(config["input_w"]):
        raise ValueError("This script expects square model input_h/input_w.")
    if args.overlap < 0 or args.overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    channels = 1 if config["num_classes"] == 1 else int(config["num_classes"])
    os.makedirs(args.temp_dir, exist_ok=True)

    with rasterio.open(args.input_tif) as src:
        band_indexes = list(config["selected_bands"])
        if src.count < max(band_indexes):
            raise ValueError("Input has %d bands, but selected_bands=%s" % (src.count, band_indexes))

        width = src.width
        height = src.height
        stride = tile_size - args.overlap
        xs = make_starts(width, tile_size, stride)
        ys = make_starts(height, tile_size, stride)
        total = len(xs) * len(ys) * len(scales)
        weight_window = build_weight_window(tile_size)

        score_path = os.path.join(args.temp_dir, "score_%s_%dx%d.dat" % (args.name, width, height))
        weight_path = os.path.join(args.temp_dir, "weight_%s_%dx%d.dat" % (args.name, width, height))
        score_sum = np.memmap(score_path, mode="w+", dtype=np.float32, shape=(channels, height, width))
        weight_sum = np.memmap(weight_path, mode="w+", dtype=np.float32, shape=(height, width))
        score_sum[:] = 0
        weight_sum[:] = 0

        print("========== big tif inference ==========")
        print("model name      :", args.name)
        print("arch            :", config["arch"])
        print("input tif       :", args.input_tif)
        print("out class       :", args.out_class)
        print("out conf        :", args.out_conf or "(disabled)")
        print("selected bands  :", band_indexes)
        print("band names      :", config.get("selected_band_names", "unknown"))
        print("image size      :", "%d x %d" % (width, height))
        print("model size      :", model_size)
        print("tile/overlap    :", "%d / %d" % (tile_size, args.overlap))
        print("context scales  :", scales)
        print("context weights :", [round(v, 4) for v in scale_weights])
        print("device          :", device)
        print("tiles total     :", total)
        print("=======================================")

        pending = []
        used = 0
        skipped = 0
        with tqdm(total=total, desc="infer", ncols=100) as pbar:
            for y0 in ys:
                for x0 in xs:
                    base_patch = src.read(
                        indexes=band_indexes,
                        window=Window(x0, y0, tile_size, tile_size),
                        boundless=True,
                        fill_value=0,
                    )
                    if args.skip_zero_tiles and should_skip_tile(base_patch, args.skip_zero_ratio):
                        skipped += len(scales)
                        pbar.update(len(scales))
                        continue

                    h = min(tile_size, height - y0)
                    w = min(tile_size, width - x0)
                    for scale, scale_weight in zip(scales, scale_weights):
                        patch, context_size, context_margin = read_context_patch(
                            src, x0, y0, tile_size, scale, band_indexes
                        )
                        tensor = prepare_patch(patch, config, model_size)
                        pending.append(
                            {
                                "tensor": tensor,
                                "x0": x0,
                                "y0": y0,
                                "h": h,
                                "w": w,
                                "tile_size": tile_size,
                                "context_size": context_size,
                                "context_margin": context_margin,
                                "scale_weight": float(scale_weight),
                            }
                        )
                        if len(pending) >= args.batch_size:
                            used += flush_pending(
                                model, pending, score_sum, weight_sum, weight_window, config, device
                            )
                        pbar.update(1)

        used += flush_pending(model, pending, score_sum, weight_sum, weight_window, config, device)
        write_outputs(src, args, config, score_sum, weight_sum, channels, height, width)

        print("[done] class map:", args.out_class)
        if args.out_conf:
            print("[done] confidence map:", args.out_conf)
        print("[info] inferred tiles:", used)
        print("[info] skipped tiles :", skipped)

    if not args.keep_temp:
        for path in [score_path, weight_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
