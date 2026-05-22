import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset, VOCDataset
from metrics import binary_confusion_counts, binary_segmentation_metrics, iou_score
from multispectral_config import (
    band_mode,
    dataset_name,
    image_ext,
    ignore_zero_pixels,
    input_channels,
    mask_ext,
    nodata_value,
    normalization_config,
    selected_band_names,
    selected_bands,
    train_augmentation_config,
    vis_bands,
)
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

#先在multispectral_config.py中选择配置的波段信息，然后在下面配置参数即可

TRAIN_DEFAULTS = {
    #'name': None,默认模型名字，如果不传命令行参数，默认是arch+timestamp
    'name': 'voc_run5',
    'epochs': 20,
    'batch_size': 2,
    'arch': 'NestedUNet',
    'deep_supervision': False,
    # 第三步修改：训练入口接入统一多光谱配置。
    # 原来这里写死 input_channels=3，只适合普通 RGB。
    # 现在从 multispectral_config.py 读取，band_mode="6band" 时自动变成 6。
    'input_channels': input_channels,
    'num_classes': 1,
    'input_w': 256,
    'input_h': 256,
    'loss': 'BCEDiceLoss',
    # 第三步修改：数据集路径和影像后缀也统一从配置文件读取。
    # 这样切换 rgb / 4band / 6band 时，训练入口不用反复手改。
    'dataset': dataset_name,
    'img_ext': image_ext,
    'mask_ext': mask_ext,
    'optimizer': 'SGD',
    'lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'nesterov': False,
    'scheduler': 'CosineAnnealingLR',
    'min_lr': 1e-5,
    'factor': 0.1,
    'patience': 2,
    'milestones': '1,2',
    'gamma': 2/3,
    'early_stopping': -1,
    'num_workers': 0,
}


def attach_multispectral_config(config):
    # 第三步补充：把论文实验需要复现的多光谱信息写进 config。
    # 后面 config 会保存到 models/<name>/config.yml，
    # 这样每次实验使用了哪些波段、哪些标准化参数都能追溯。
    config['band_mode'] = band_mode
    config['selected_bands'] = list(selected_bands)
    config['selected_band_names'] = list(selected_band_names)
    config['vis_bands'] = list(vis_bands)
    config['nodata_value'] = nodata_value
    config['ignore_zero_pixels'] = ignore_zero_pixels
    config['normalization_config'] = normalization_config
    config['train_augmentation_config'] = train_augmentation_config
    # 第五步修改：让模型输入通道数始终等于当前选择的波段数。
    # 即使命令行误传了 --input_channels，也以 multispectral_config.py 为准，
    # 避免模型第一层通道数和 dataset.py 实际输出通道数不一致。
    config['input_channels'] = len(config['selected_bands'])
    return config


def build_transforms(config):
    # 第三步补充：区分 RGB 增强和多光谱增强。
    # HueSaturationValue 是 RGB/HSV 颜色空间增强，不适合 4/6 波段遥感反射率。
    # 多光谱模式下只保留几何增强，避免破坏各波段的物理含义。
    # Random training augmentation is handled in dataset.py through
    # train_augmentation_config so it stays aligned with the u2net branch.
    train_transforms = [
        A.Resize(config['input_h'], config['input_w']),
    ]

    # 只有普通 jpg/png 的 RGB 图像才保留颜色增强。
    # 如果 band_mode="rgb" 但 img_ext=".tif"，它仍然是 Sentinel-2 反射率数据，
    # 不适合使用 HSV 这类面向自然图像的颜色扰动。
    # 第三步补充：去掉 A.Normalize()。
    # 原因是 dataset.py 已经统一完成：
    # - RGB: /255
    # - tif: /10000 -> clip -> mean/std
    # 如果这里再调用 Albumentations 默认 Normalize，会把输入重复标准化，
    # 且默认 ImageNet RGB 参数不适合多光谱。
    train_transform = A.Compose(train_transforms)
    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
    ])

    return train_transform, val_transform

"""
直接在上面的 TRAIN_DEFAULTS 里改训练参数。
如果不再传命令行参数，运行：
python train.py
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=TRAIN_DEFAULTS['name'],
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=TRAIN_DEFAULTS['epochs'], type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=TRAIN_DEFAULTS['batch_size'], type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default=TRAIN_DEFAULTS['arch'],
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=TRAIN_DEFAULTS['deep_supervision'], type=str2bool)
    parser.add_argument('--input_channels', default=TRAIN_DEFAULTS['input_channels'], type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=TRAIN_DEFAULTS['num_classes'], type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=TRAIN_DEFAULTS['input_w'], type=int,
                        help='image width')
    parser.add_argument('--input_h', default=TRAIN_DEFAULTS['input_h'], type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default=TRAIN_DEFAULTS['loss'],
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default=TRAIN_DEFAULTS['dataset'],
                        help='dataset name')
    parser.add_argument('--img_ext', default=TRAIN_DEFAULTS['img_ext'],
                        help='image file extension')
    parser.add_argument('--mask_ext', default=TRAIN_DEFAULTS['mask_ext'],
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default=TRAIN_DEFAULTS['optimizer'],
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=TRAIN_DEFAULTS['lr'], type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=TRAIN_DEFAULTS['momentum'], type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=TRAIN_DEFAULTS['weight_decay'], type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=TRAIN_DEFAULTS['nesterov'], type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default=TRAIN_DEFAULTS['scheduler'],
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=TRAIN_DEFAULTS['min_lr'], type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=TRAIN_DEFAULTS['factor'], type=float)
    parser.add_argument('--patience', default=TRAIN_DEFAULTS['patience'], type=int)
    parser.add_argument('--milestones', default=TRAIN_DEFAULTS['milestones'], type=str)
    parser.add_argument('--gamma', default=TRAIN_DEFAULTS['gamma'], type=float)
    parser.add_argument('--early_stopping', default=TRAIN_DEFAULTS['early_stopping'], type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=TRAIN_DEFAULTS['num_workers'], type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            tp, fp, fn, tn = binary_confusion_counts(
                outputs[-1] if config['deep_supervision'] else output,
                target,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    metrics = binary_segmentation_metrics(total_tp, total_fp, total_fn, total_tn)

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('batch_mean_iou', avg_meters['iou'].avg),
        ('iou', metrics['iou']),
        ('miou', metrics['miou']),
        ('precision', metrics['precision']),
        ('recall', metrics['recall']),
        ('f1', metrics['f1']),
    ])


def get_dataset_info(config):
    default_root = os.path.join('inputs', config['dataset'])
    voc_root = config['dataset']

    if os.path.isdir(os.path.join(default_root, 'images')) and os.path.isdir(os.path.join(default_root, 'masks')):
        img_ids = glob(os.path.join(default_root, 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        return {
            'dataset_class': Dataset,
            'train_img_ids': train_img_ids,
            'val_img_ids': val_img_ids,
            'kwargs': {
                'img_dir': os.path.join(default_root, 'images'),
                'mask_dir': os.path.join(default_root, 'masks'),
                'img_ext': config['img_ext'],
                'mask_ext': config['mask_ext'],
                'num_classes': config['num_classes'],
                # 第三步补充：把波段选择和标准化配置传给 dataset.py。
                # 真正读取哪些 tif 波段，是在数据集对象里执行的。
                'selected_bands': config['selected_bands'],
                'normalization_config': config['normalization_config'],
            },
        }

    voc_img_dir = os.path.join(voc_root, 'JPEGImages')
    voc_mask_dir = os.path.join(voc_root, 'SegmentationClass')
    voc_split_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
    train_split = os.path.join(voc_split_dir, 'train.txt')
    val_split = os.path.join(voc_split_dir, 'val.txt')

    if os.path.isdir(voc_img_dir) and os.path.isdir(voc_mask_dir):
        if os.path.isfile(train_split) and os.path.isfile(val_split):
            with open(train_split, 'r') as f:
                train_img_ids = [line.strip() for line in f if line.strip()]
            with open(val_split, 'r') as f:
                val_img_ids = [line.strip() for line in f if line.strip()]
        else:
            # 第三步修改：VOC fallback 不再写死查找 *.jpg。
            # 多光谱数据通常仍放在 JPEGImages 目录下，但后缀是 .tif。
            img_ids = glob(os.path.join(voc_img_dir, '*' + config['img_ext']))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        return {
            'dataset_class': VOCDataset,
            'train_img_ids': train_img_ids,
            'val_img_ids': val_img_ids,
            'kwargs': {
                'img_dir': voc_img_dir,
                'mask_dir': voc_mask_dir,
                # 第三步修改：VOC 数据集也使用统一配置里的后缀。
                # 原来这里固定 .jpg/.png，会导致 tif 训练时找不到影像。
                'img_ext': config['img_ext'],
                'mask_ext': config['mask_ext'],
                'selected_bands': config['selected_bands'],
                'normalization_config': config['normalization_config'],
            },
        }

    raise FileNotFoundError(
        'Dataset not found. Expected either inputs/<dataset>/images + masks, '
        'or a VOC-style directory with JPEGImages and SegmentationClass.'
    )


def main():
    config = attach_multispectral_config(vars(parse_args()))

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()#WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    dataset_info = get_dataset_info(config)
    train_transform, val_transform = build_transforms(config)

    train_dataset = dataset_info['dataset_class'](
        img_ids=dataset_info['train_img_ids'],
        transform=train_transform,
        augmentation_config=config['train_augmentation_config'],
        **dataset_info['kwargs'])
    val_dataset = dataset_info['dataset_class'](
        img_ids=dataset_info['val_img_ids'],
        transform=val_transform,
        **dataset_info['kwargs'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)#不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_miou', []),
        ('val_precision', []),
        ('val_recall', []),
        ('val_f1', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(
            'loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_miou %.4f - val_p %.4f - val_r %.4f - val_f1 %.4f'
            % (
                train_log['loss'],
                train_log['iou'],
                val_log['loss'],
                val_log['iou'],
                val_log['miou'],
                val_log['precision'],
                val_log['recall'],
                val_log['f1'],
            )
        )

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_miou'].append(val_log['miou'])
        log['val_precision'].append(val_log['precision'])
        log['val_recall'].append(val_log['recall'])
        log['val_f1'].append(val_log['f1'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
