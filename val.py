import argparse
import os
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
import csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset, VOCDataset
from metrics import iou_score
from train import TRAIN_DEFAULTS
from multispectral_config import (
    normalization_config,
    selected_bands,
    vis_bands,
)
from utils import AverageMeter

VAL_DEFAULTS = {
    'name': TRAIN_DEFAULTS['name'],
    # 第九步修改：把验证/测试 split 也放到文件顶部默认配置里。
    # 以后不想写命令行参数时，直接在这里改：
    # 在这里选择验证集还是测试集
    # - 'val'  -> 读取 ImageSets/Segmentation/val.txt
    # - 'test' -> 读取 ImageSets/Segmentation/test.txt
    'split': 'val',
}


def complete_multispectral_config(config):
    # 第七步复核补充：兼容旧的 config.yml。
    # 新训练会保存 selected_bands / normalization_config / vis_bands；
    # 但如果拿旧模型跑 val.py，这些字段可能不存在。
    # 这里补默认值，保证验证入口和 dataset.py 的多光谱读取参数完整。
    config.setdefault('selected_bands', list(selected_bands))
    config.setdefault('normalization_config', normalization_config)
    config.setdefault('vis_bands', list(vis_bands))
    config['input_channels'] = len(config['selected_bands'])
    return config

"""
直接在上面的 VAL_DEFAULTS 里改验证参数。
如果不再传命令行参数，运行：
python val.py
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=VAL_DEFAULTS['name'],
                        help='model name')
    parser.add_argument(
        '--split',
        default=VAL_DEFAULTS['split'],
        choices=['val', 'test'],
        help='VOC split to evaluate: val or test',
    )

    args = parser.parse_args()

    return args


def get_output_name(model_name, split):
    # 第八步修改：验证集和测试集结果分开保存。
    # 默认 val 仍然输出到 outputs/<name>，兼容前面的使用习惯；
    # test 输出到 outputs/<name>_test，避免覆盖验证集 metrics 和预测图。
    if split == 'val':
        return model_name
    return '%s_%s' % (model_name, split)


def fast_hist(label, pred, num_classes):
    valid = (label >= 0) & (label < num_classes)
    return np.bincount(
        num_classes * label[valid].astype(int) + pred[valid].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def per_class_iou(hist):
    return np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1)


def per_class_recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def pixel_accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def format_percent(value):
    return '%.3f' % (float(value) * 100)


def compute_hist_metrics(hist):
    ious = per_class_iou(hist)
    recalls = per_class_recall(hist)
    precisions = per_class_precision(hist)
    return {
        'hist': hist,
        'ious': ious,
        'recalls': recalls,
        'precisions': precisions,
        'miou': np.nanmean(ious),
        'mpa': np.nanmean(recalls),
        'mprecision': np.nanmean(precisions),
        'accuracy': pixel_accuracy(hist),
    }


def print_miou_metrics(metrics, name_classes):
    for class_index, class_name in enumerate(name_classes):
        print(
            '===>%s:\tIoU-%s; Recall (equal to the PA)-%s; Precision-%s'
            % (
                class_name,
                format_percent(metrics['ious'][class_index]),
                format_percent(metrics['recalls'][class_index]),
                format_percent(metrics['precisions'][class_index]),
            )
        )
    print(
        '===> mIoU: %s; mPA: %s; Accuracy: %s; mPrecision: %s'
        % (
            format_percent(metrics['miou']),
            format_percent(metrics['mpa']),
            format_percent(metrics['accuracy']),
            format_percent(metrics['mprecision']),
        )
    )


def adjust_axes(renderer, text, fig, axes):
    bbox = text.get_window_extent(renderer=renderer)
    text_width_inches = bbox.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12):
    fig = plt.figure()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)

    renderer = fig.canvas.get_renderer()
    for index, value in enumerate(values):
        text_value = ' ' + str(round(float(value), 3))
        if value < 1.0:
            text_value = ' %.3f' % float(value)
        text = plt.text(
            value,
            index,
            text_value,
            color='royalblue',
            va='center',
            fontweight='bold',
        )
        if index == len(values) - 1:
            adjust_axes(renderer, text, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def show_results(miou_out_path, metrics, name_classes, tick_font_size=12):
    os.makedirs(miou_out_path, exist_ok=True)

    draw_plot_func(
        metrics['ious'],
        name_classes,
        'mIoU = %.3f%%' % (metrics['miou'] * 100),
        'Intersection over Union',
        os.path.join(miou_out_path, 'mIoU.png'),
        tick_font_size=tick_font_size,
    )
    print('Save mIoU out to ' + os.path.join(miou_out_path, 'mIoU.png'))

    draw_plot_func(
        metrics['recalls'],
        name_classes,
        'mPA = %.3f%%' % (metrics['mpa'] * 100),
        'Pixel Accuracy',
        os.path.join(miou_out_path, 'mPA.png'),
        tick_font_size=tick_font_size,
    )
    print('Save mPA out to ' + os.path.join(miou_out_path, 'mPA.png'))

    draw_plot_func(
        metrics['recalls'],
        name_classes,
        'mRecall = %.3f%%' % (metrics['mpa'] * 100),
        'Recall',
        os.path.join(miou_out_path, 'Recall.png'),
        tick_font_size=tick_font_size,
    )
    print('Save Recall out to ' + os.path.join(miou_out_path, 'Recall.png'))

    draw_plot_func(
        metrics['precisions'],
        name_classes,
        'mPrecision = %.3f%%' % (metrics['mprecision'] * 100),
        'Precision',
        os.path.join(miou_out_path, 'Precision.png'),
        tick_font_size=tick_font_size,
    )
    print('Save Precision out to ' + os.path.join(miou_out_path, 'Precision.png'))

    confusion_matrix_path = os.path.join(miou_out_path, 'confusion_matrix.csv')
    with open(confusion_matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + [str(c) for c in name_classes])
        for class_index, class_name in enumerate(name_classes):
            writer.writerow([class_name] + [str(int(x)) for x in metrics['hist'][class_index]])
    print('Save confusion_matrix out to ' + confusion_matrix_path)


def get_dataset_info(config, split='val'):
    # 第八步修改：val.py 支持选择 val.txt 或 test.txt。
    # 论文实验里通常用 val.txt 调参/选模型，用 test.txt 做最终报告。
    # 因此这里把 split 显式传进来，而不是固定读取 val.txt。
    default_root = os.path.join('inputs', config['dataset'])
    voc_roots = [
        config['dataset'],
        os.path.join('VOCdevkit', config['dataset']),
    ]

    if os.path.isdir(os.path.join(default_root, 'images')) and os.path.isdir(os.path.join(default_root, 'masks')):
        img_ids = glob(os.path.join(default_root, 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        if split != 'val':
            raise FileNotFoundError(
                "inputs/<dataset>/images format has no %s.txt split file. "
                "Use VOC format if you want to evaluate test.txt." % split
            )
        _, eval_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        return {
            'dataset_class': Dataset,
            'eval_img_ids': eval_img_ids,
            'kwargs': {
                'img_dir': os.path.join(default_root, 'images'),
                'mask_dir': os.path.join(default_root, 'masks'),
                'img_ext': config['img_ext'],
                'mask_ext': config['mask_ext'],
                'num_classes': config['num_classes'],
                # 第四步修改：验证入口也要把训练时保存的多光谱配置传给 Dataset。
                # 否则 val.py 单独运行时，虽然模型是 6 通道，数据读取却不知道该选哪些波段。
                'selected_bands': config.get('selected_bands'),
                'normalization_config': config.get('normalization_config'),
            },
        }

    for voc_root in voc_roots:
        voc_img_dir = os.path.join(voc_root, 'JPEGImages')
        voc_mask_dir = os.path.join(voc_root, 'SegmentationClass')
        voc_split_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
        split_path = os.path.join(voc_split_dir, split + '.txt')

        if os.path.isdir(voc_img_dir) and os.path.isdir(voc_mask_dir):
            if os.path.isfile(split_path):
                with open(split_path, 'r') as f:
                    eval_img_ids = [line.strip() for line in f if line.strip()]
            else:
                if split != 'val':
                    raise FileNotFoundError("Split file not found: %s" % split_path)
                # 第四步修改：VOC 验证 fallback 不再写死查找 *.jpg。
                # 多光谱 VOC 结构仍可能使用 JPEGImages 目录名，但影像后缀是 .tif。
                img_ids = glob(os.path.join(voc_img_dir, '*' + config['img_ext']))
                img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
                _, eval_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

            return {
                'dataset_class': VOCDataset,
                'eval_img_ids': eval_img_ids,
                'kwargs': {
                    'img_dir': voc_img_dir,
                    'mask_dir': voc_mask_dir,
                    # 第四步修改：这里必须沿用训练 config 中的后缀和预处理配置。
                    # 原来固定 .jpg/.png，会导致 6 波段 tif 验证时找不到输入图。
                    'img_ext': config['img_ext'],
                    'mask_ext': config['mask_ext'],
                    'selected_bands': config.get('selected_bands'),
                    'normalization_config': config.get('normalization_config'),
                },
            }

    raise FileNotFoundError(
        'Dataset not found. Expected either inputs/<dataset>/images + masks, '
        'or a VOC-style directory with JPEGImages and SegmentationClass.'
    )


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = complete_multispectral_config(config)
    run_name = args.name
    config['eval_split'] = args.split
    output_name = get_output_name(run_name, args.split)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('%s: %s' % ('run_name', run_name))
    print('%s: %s' % ('output_name', output_name))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    dataset_info = get_dataset_info(config, args.split)

    checkpoint_path = os.path.join('models', run_name, 'model.pth')
    print('=> loading checkpoint %s' % checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 第四步修改：去掉验证阶段的 A.Normalize()。
    # train.py / dataset.py 已经统一完成 RGB /255 或 tif /10000+clip+mean_std。
    # 验证阶段只做尺寸对齐，避免和训练预处理不一致。
    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
    ])

    val_dataset = dataset_info['dataset_class'](
        img_ids=dataset_info['eval_img_ids'],
        transform=val_transform,
        **dataset_info['kwargs'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    metric_num_classes = 2 if config['num_classes'] == 1 else config['num_classes']
    name_classes = ['_background_', 'PV'] if metric_num_classes == 2 else [
        'class_%d' % i for i in range(metric_num_classes)
    ]
    hist = np.zeros((metric_num_classes, metric_num_classes), dtype=np.float64)

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', output_name, str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))
            if config['num_classes'] == 1:
                pred = (torch.sigmoid(output) > 0.5).long().squeeze(1).cpu().numpy()
                label = (target > 0.5).long().squeeze(1).cpu().numpy()
            else:
                pred = torch.argmax(output, dim=1).cpu().numpy()
                label = torch.argmax(target, dim=1).cpu().numpy()

            for label_i, pred_i in zip(label, pred):
                hist += fast_hist(label_i.flatten(), pred_i.flatten(), metric_num_classes)

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', output_name, str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    metrics = compute_hist_metrics(hist)

    print('BatchMeanIoU: %.3f' % (avg_meter.avg * 100))
    print_miou_metrics(metrics, name_classes)

    save_metrics(output_name, avg_meter.avg, metrics, name_classes, args.split)
    show_results(os.path.join('miou_out', output_name), metrics, name_classes)

    plot_examples(input, target, model, config, output_name, num_examples=3)
    
    torch.cuda.empty_cache()

def image_for_display(image_chw, config):
    # 第四步补充：多光谱输入不能直接当 RGB 显示。
    # 这里仅用于 examples.png 可视化：
    # - 3 通道时直接取前三通道
    # - 4/6 通道时按 selected_bands 找 B4/B3/B2 对应的通道
    # - 如果找不到，就退回显示第一个通道
    image = np.transpose(image_chw.cpu().numpy(), (1, 2, 0))
    selected = config.get('selected_bands') or []
    vis_bands = config.get('vis_bands', [3, 2, 1])

    if image.shape[-1] >= 3:
        if all(b in selected for b in vis_bands):
            channel_indexes = [selected.index(b) for b in vis_bands]
            image = image[:, :, channel_indexes]
        else:
            image = image[:, :, :3]
        image_min = np.percentile(image, 2)
        image_max = np.percentile(image, 98)
        return np.clip((image - image_min) / max(image_max - image_min, 1e-6), 0, 1)

    return image[:, :, 0]


def plot_examples(datax, datay, model, config, output_name, num_examples=6):
    m = datax.shape[0]
    num_examples = min(num_examples, m)
    if num_examples <= 0:
        return
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    if num_examples == 1:
        ax = np.expand_dims(ax, axis=0)
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(image_for_display(datax[image_indx], config))
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    save_path = os.path.join('outputs', output_name, 'examples.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def rounded_percent(value):
    return round(float(value) * 100, 3)


def save_metrics(run_name, batch_mean_iou, metrics, name_classes, split):
    save_dir = os.path.join('outputs', run_name)
    os.makedirs(save_dir, exist_ok=True)

    yaml_path = os.path.join(save_dir, 'metrics.yml')
    csv_path = os.path.join(save_dir, 'metrics.csv')
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.csv')

    payload = {
        'split': split,
        'batch_mean_iou_percent': rounded_percent(batch_mean_iou),
        'miou_percent': rounded_percent(metrics['miou']),
        'mpa_percent': rounded_percent(metrics['mpa']),
        'accuracy_percent': rounded_percent(metrics['accuracy']),
        'mprecision_percent': rounded_percent(metrics['mprecision']),
        'classes': {},
    }
    for class_index, class_name in enumerate(name_classes):
        payload['classes'][class_name] = {
            'iou_percent': rounded_percent(metrics['ious'][class_index]),
            'recall_pa_percent': rounded_percent(metrics['recalls'][class_index]),
            'precision_percent': rounded_percent(metrics['precisions'][class_index]),
        }

    with open(yaml_path, 'w') as f:
        yaml.dump(payload, f)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key in ['split', 'batch_mean_iou_percent', 'miou_percent', 'mpa_percent', 'accuracy_percent', 'mprecision_percent']:
            writer.writerow([key, payload[key]])
        writer.writerow([])
        writer.writerow(['class', 'iou_percent', 'recall_pa_percent', 'precision_percent'])
        for class_name in name_classes:
            class_metrics = payload['classes'][class_name]
            writer.writerow([
                class_name,
                class_metrics['iou_percent'],
                class_metrics['recall_pa_percent'],
                class_metrics['precision_percent'],
            ])

    with open(confusion_matrix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + [str(c) for c in name_classes])
        for class_index, class_name in enumerate(name_classes):
            writer.writerow([class_name] + [str(int(x)) for x in metrics['hist'][class_index]])


if __name__ == '__main__':
    main()
