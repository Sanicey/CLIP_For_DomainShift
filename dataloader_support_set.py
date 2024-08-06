import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from  datasets.bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from datasets.sampler import RandomIdentitySampler
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from PIL import Image
import os
from torchvision.transforms import Compose, Resize, ToTensor
import random
from torchvision import datasets, transforms

'''
(4) dataloader_support_set.py
different functions for dataloader
'''

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
}

def get_source_domain_param(cfg):
    print("Load source domain's param: ", cfg.DATASETS.NAMES)

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    return num_classes, cam_num, view_num

def get_target_domain_train_imgs_for_support_set(cfg, num_of_splits):
    target_domain_dataset = __factory[cfg.Target.NAMES](root=cfg.Target.ROOT_DIR)

    # 修改预处理函数，以适应多张图片
    preprocess = Compose([
        transforms.Resize((256, 128)), # 调整图片大小
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，这里使用的是ImageNet的均值和方差
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 在建立support_set的时候，可以不用normalize？
    ])

    # target_domain_dataset.traindir 目标图片所在文件夹地址
    train_dir = target_domain_dataset.train_dir
    image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images = [Image.open(img_path) for img_path in image_paths]

    # 将图片分割成等大小的列表
    split_images = []
    split_size = len(images) // num_of_splits
    remainder = len(images) % num_of_splits

    start = 0
    for i in range(num_of_splits):
        end = start + split_size
        if i < remainder:
            end += 1
        split_images.append(images[start:end])
        start = end

    # 预处理图片，每一组内部拼接在一起
    target_domain_all_images = [torch.stack([preprocess(img) for img in img_list]) for img_list in split_images] # 100 * (130,3,256,128)

    return target_domain_all_images, image_paths

def get_support_set_data(cfg, num_support_set_augmentation):
    # 数据预处理，要不要做数据扩充？
    data_transforms = transforms.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),

        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),

        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # 加载数据集
    dataset_dir = cfg.Support_set.ROOT_DIR + "/" + cfg.Support_set.NAMES  # 请替换为support_set所在的文件夹路径
    image_dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
    # 这里的cam_num放99防止计算map的时候出问题
    dataset_train = [(image, int(label), 99, 0) for image, label in image_dataset]  # cam_num, view_num

    # 数据复制 + 增强
    dataset_train = dataset_train * num_support_set_augmentation
    dataset_train = sorted(dataset_train, key=lambda x: x[1])

    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                             sampler=RandomIdentitySampler(dataset_train,
                                                                           cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                                           cfg.DATALOADER.NUM_INSTANCE),
                                             num_workers=num_workers)

    return dataloader

def get_val_data(cfg):
    def val_collate_fn(batch):
        # 整理验证数据批次，返回图像、pids、camids、camids_batch（可能用于不同的用途）、viewids和图像路径
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.Target.NAMES](root=cfg.Target.ROOT_DIR)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return val_loader, len(dataset.query)

