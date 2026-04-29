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
from metrics import binary_confusion_counts, binary_segmentation_metrics, iou_score
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
    'split': 'test',
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
    config['eval_split'] = args.split
    output_name = get_output_name(config['name'], args.split)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
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

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
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
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

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
            tp, fp, fn, tn = binary_confusion_counts(output, target)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', output_name, str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    metrics = binary_segmentation_metrics(total_tp, total_fp, total_fn, total_tn)

    print('BatchMeanIoU: %.4f' % avg_meter.avg)
    print('Precision: %.4f' % metrics['precision'])
    print('Recall: %.4f' % metrics['recall'])
    print('F1: %.4f' % metrics['f1'])
    print('IoU: %.4f' % metrics['iou'])
    print('mIoU: %.4f' % metrics['miou'])

    save_metrics(output_name, avg_meter.avg, metrics, total_tp, total_fp, total_fn, total_tn, args.split)

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


def save_metrics(run_name, batch_mean_iou, metrics, tp, fp, fn, tn, split):
    save_dir = os.path.join('outputs', run_name)
    os.makedirs(save_dir, exist_ok=True)

    yaml_path = os.path.join(save_dir, 'metrics.yml')
    csv_path = os.path.join(save_dir, 'metrics.csv')

    payload = {
        'split': split,
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'iou': float(metrics['iou']),
        'miou': float(metrics['miou']),
        'batch_mean_iou': float(batch_mean_iou),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(payload, f)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key in ['split', 'precision', 'recall', 'f1', 'iou', 'miou', 'batch_mean_iou', 'tp', 'fp', 'fn', 'tn']:
            writer.writerow([key, payload[key]])


if __name__ == '__main__':
    main()
