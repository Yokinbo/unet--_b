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
from utils import AverageMeter

VAL_DEFAULTS = {
    'name': TRAIN_DEFAULTS['name'],
}

"""
直接在上面的 VAL_DEFAULTS 里改验证参数。
如果不再传命令行参数，运行：
python val.py
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=VAL_DEFAULTS['name'],
                        help='model name')

    args = parser.parse_args()

    return args


def get_dataset_info(config):
    default_root = os.path.join('inputs', config['dataset'])
    voc_roots = [
        config['dataset'],
        os.path.join('VOCdevkit', config['dataset']),
    ]

    if os.path.isdir(os.path.join(default_root, 'images')) and os.path.isdir(os.path.join(default_root, 'masks')):
        img_ids = glob(os.path.join(default_root, 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

        return {
            'dataset_class': Dataset,
            'val_img_ids': val_img_ids,
            'kwargs': {
                'img_dir': os.path.join(default_root, 'images'),
                'mask_dir': os.path.join(default_root, 'masks'),
                'img_ext': config['img_ext'],
                'mask_ext': config['mask_ext'],
                'num_classes': config['num_classes'],
            },
        }

    for voc_root in voc_roots:
        voc_img_dir = os.path.join(voc_root, 'JPEGImages')
        voc_mask_dir = os.path.join(voc_root, 'SegmentationClass')
        voc_split_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
        val_split = os.path.join(voc_split_dir, 'val.txt')

        if os.path.isdir(voc_img_dir) and os.path.isdir(voc_mask_dir):
            if os.path.isfile(val_split):
                with open(val_split, 'r') as f:
                    val_img_ids = [line.strip() for line in f if line.strip()]
            else:
                img_ids = glob(os.path.join(voc_img_dir, '*.jpg'))
                img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
                _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

            return {
                'dataset_class': VOCDataset,
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
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    dataset_info = get_dataset_info(config)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = dataset_info['dataset_class'](
        img_ids=dataset_info['val_img_ids'],
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
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
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
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    metrics = binary_segmentation_metrics(total_tp, total_fp, total_fn, total_tn)

    print('BatchMeanIoU: %.4f' % avg_meter.avg)
    print('Precision: %.4f' % metrics['precision'])
    print('Recall: %.4f' % metrics['recall'])
    print('F1: %.4f' % metrics['f1'])
    print('IoU: %.4f' % metrics['iou'])
    print('mIoU: %.4f' % metrics['miou'])

    save_metrics(config['name'], avg_meter.avg, metrics, total_tp, total_fp, total_fn, total_tn)

    plot_examples(input, target, model, config['name'], num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model, run_name, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    save_path = os.path.join('outputs', run_name, 'examples.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_metrics(run_name, batch_mean_iou, metrics, tp, fp, fn, tn):
    save_dir = os.path.join('outputs', run_name)
    os.makedirs(save_dir, exist_ok=True)

    yaml_path = os.path.join(save_dir, 'metrics.yml')
    csv_path = os.path.join(save_dir, 'metrics.csv')

    payload = {
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
        for key in ['precision', 'recall', 'f1', 'iou', 'miou', 'batch_mean_iou', 'tp', 'fp', 'fn', 'tn']:
            writer.writerow([key, payload[key]])


if __name__ == '__main__':
    main()
