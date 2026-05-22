"""
Unified configuration for RGB and Sentinel-2 multispectral experiments.

Keep training, validation, and future inference on the same band selection
and normalization settings. Change band_mode first when switching experiments.
"""

# Dataset / image format.
dataset_name = "VOCdevkit/VOC2007"
image_ext = ".tif"
mask_ext = ".png"

# Supported modes:
# "rgb"   -> Sentinel-2 true color: [B4, B3, B2]
# "4band" -> visible + NIR:       [B2, B3, B4, B8]
# "6band" -> visible + NIR + SWIR: [B2, B3, B4, B8, B11, B12]
band_mode = "rgb"

# The source 6-band tif order is assumed to be:
# [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
band_options = {
    "rgb": [3, 2, 1],
    "4band": [1, 2, 3, 4],
    "6band": [1, 2, 3, 4, 5, 6],
}

band_names = {
    "rgb": ["B4", "B3", "B2"],
    "4band": ["B2", "B3", "B4", "B8"],
    "6band": ["B2", "B3", "B4", "B8", "B11", "B12"],
}

selected_bands = band_options[band_mode]
selected_band_names = band_names[band_mode]
input_channels = len(selected_bands)

# Visualization bands are 1-based indexes in the source 6-band tif.
# B4/B3/B2 gives a true-color preview.
vis_bands = [3, 2, 1]

# NoData / invalid-pixel handling for statistics and optional masking.
nodata_value = 0
ignore_zero_pixels = True


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
# Sentinel-2 optical bands are commonly stored as quantized reflectance:
# reflectance ~= DN / 10000. Statistics below are computed after this scale.
#
# For paper-grade experiments, compute clip/mean/std from the training split
# only. Do not use validation or test pixels to avoid data leakage.
# ---------------------------------------------------------------------
normalization_configs = {
    "rgb": {
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        # Channel order: [B4, B3, B2]
        "clip_min": [0.029400, 0.044100, 0.024300],
        "clip_max": [0.286500, 0.228700, 0.207400],
        "enable_mean_std": True,
        "mean": [0.130562, 0.108342, 0.076092],
        "std": [0.053213, 0.038353, 0.034616],
    },
    "4band": {
        "reflectance_scale": 10000.0,
        # Channel order: [B2, B3, B4, B8]
        "enable_clip": True,
        "clip_min": [0.024300, 0.044100, 0.029400, 0.121600],
        "clip_max": [0.207400, 0.228700, 0.286500, 0.447600],
        "enable_mean_std": True,
        "mean": [0.076092, 0.108342, 0.130562, 0.243092],
        "std": [0.034616, 0.038353, 0.053213, 0.058950],
    },
    "6band": {
        "reflectance_scale": 10000.0,
        # Channel order: [B2, B3, B4, B8, B11, B12]
        "enable_clip": True,
        "clip_min": [0.024300, 0.044100, 0.029400, 0.121600, 0.142000, 0.084200],
        "clip_max": [0.207400, 0.228700, 0.286500, 0.447600, 0.429700, 0.402300],
        "enable_mean_std": True,
        "mean": [0.076092, 0.108342, 0.130562, 0.243092, 0.284875, 0.234912],
        "std": [0.034616, 0.038353, 0.053213, 0.058950, 0.059411, 0.065606],
    },
}

normalization_config = normalization_configs[band_mode]


# ---------------------------------------------------------------------
# Training-time online data augmentation
# ---------------------------------------------------------------------
# Keep this block aligned with the u2net branch for fair model comparison.
# These augmentations are applied to training samples only. Validation and test
# samples remain unchanged.
train_augmentation_config = {
    "enabled": True,

    # Multispectral reflectance perturbation.
    "reflectance_prob": 0.50,
    "reflectance_global_range": [0.90, 1.10],
    "reflectance_band_range": [0.95, 1.05],

    # Geometry perturbation: hflip, vflip, rot90, rot180, rot270.
    "geometry_prob": 0.50,

    # Soft local shadow / thin cloud-shadow perturbation.
    "shadow_prob": 0.25,
    "shadow_factor_range": [0.75, 0.90],
    "shadow_radius_range": [0.25, 0.45],

    # Mild Gaussian noise.
    "noise_prob": 0.25,
    "noise_sigma_range": [0.003, 0.008],

    # Random scale by crop and resize back.
    "scale_prob": 0.20,
    "scale_crop_range": [0.85, 1.00],
}
