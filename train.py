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
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

TRAIN_DEFAULTS = {
    #'name': None,默认模型名字，如果不传命令行参数，默认是arch+timestamp
    'name': 'voc_run2',
    'epochs': 100,
    'batch_size': 2,
    'arch': 'NestedUNet',
    'deep_supervision': False,
    'input_channels': 3,
    'num_classes': 1,
    'input_w': 512,
    'input_h': 512,
    'loss': 'BCEDiceLoss',
    'dataset': 'VOCdevkit/VOC2007',
    'img_ext': '.jpg',
    'mask_ext': '.png',
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
            img_ids = glob(os.path.join(voc_img_dir, '*.jpg'))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        return {
            'dataset_class': VOCDataset,
            'train_img_ids': train_img_ids,
            'val_img_ids': val_img_ids,
            'kwargs': {
                'img_dir': voc_img_dir,
                'mask_dir': voc_mask_dir,
                'img_ext': '.jpg',
                'mask_ext': '.png',
            },
        }

    raise FileNotFoundError(
        'Dataset not found. Expected either inputs/<dataset>/images + masks, '
        'or a VOC-style directory with JPEGImages and SegmentationClass.'
    )


def main():
    config = vars(parse_args())

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
    #数据增强：
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ], p=0.5),
        A.OneOf([
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
        ], p=1),#按照归一化的概率选择执行哪一个
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = dataset_info['dataset_class'](
        img_ids=dataset_info['train_img_ids'],
        transform=train_transform,
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
