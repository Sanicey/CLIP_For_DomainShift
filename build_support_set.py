# -*- coding: utf-8 -*-
import torch
import clip
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
import os
from config import cfg
import argparse
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger
import torch.nn as nn
from torch.cuda import amp
from PIL import Image
from dataloader_support_set import get_source_domain_param, get_target_domain_train_imgs_for_support_set

'''
(2) build_support_set.py
input: 1. trained model from stage1
    2. all the tarin images from target domain
outpout: support (labels from source domain, images from target domain)
process: 1. get parameters from source domain
		2. build model, then loadparm form the trained model of stage1
		3. read all the imgs from target tarin and get the image_features
		4. take out the text_features from model:
			for text_feature in text_features:
				5. calculate the cosine similatity between image_features and text_features
				6. save the text_features(in the form of labels; source domain) and images(target domain)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the support_set")
    parser.add_argument("--config_file", default="configs/person/support_set.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 1. get parameters from source domain
    num_classes, camera_num, view_num = get_source_domain_param(cfg)

    # 2. build model, then loadparm form the trained model of stage1
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.Support_set.Stage1model)
    # for key, value in model.state_dict().items():
    #     print(key,"： ", value)
    model.eval()
    model.to("cuda")

    # 3. read all the images from target tarin and get the img_features
    num_of_splits = 100
    target_domain_all_images, image_paths = get_target_domain_train_imgs_for_support_set(cfg, num_of_splits)

    image_features_list = []

    for imgs in target_domain_all_images:
        imgs = imgs.to("cuda")
        with torch.no_grad():
            image_features = model(x=imgs, get_image=True)
            for i in image_features:
                image_features_list.append(i.cpu())

    image_features_list = torch.stack(image_features_list, dim=0)  # tensor:(16522,512)

    # 设置每个类别存下的样本数量
    num_label = 2

    # 创建文件夹
    folder_name = "./support_set/" + "support_set_of_" + cfg.Target.NAMES + "_" + str(num_label) + "_imgs_from_source_domain_" + cfg.DATASETS.NAMES
    os.makedirs(folder_name, exist_ok=True)

    # for text_feature in text_features:
    for i in range(num_classes):
        # 4. take out the text_features from model:
        text_feature = model(label = torch.tensor([i]).to("cuda"), get_text = True).to("cpu")

        # 5. calculate the cosine similarity between image_features and text_features
        similarity = (image_features_list @ text_feature.t()).t().softmax(dim=-1)

        values, indices = torch.topk(similarity[0], num_label)  # 返回输入张量中k个最大值及其对应的索引

        # 创建子文件夹
        sub_folder_name = os.path.join(folder_name, str(i))
        os.makedirs(sub_folder_name, exist_ok=True)

        # 保存图片
        for index in indices:
            img_path = image_paths[index]
            img = Image.open(img_path)
            img.save(os.path.join(sub_folder_name, os.path.basename(img_path)))











