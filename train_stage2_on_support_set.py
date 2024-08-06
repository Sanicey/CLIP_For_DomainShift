from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor_stage2_support_set import do_train_stage2
from dataloader_support_set import get_source_domain_param, get_support_set_data, get_val_data
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

'''
(3) train_stage2_on_support_set.py
input: 1. trained model from stage1(labels to text_features)
	  2. label and image from support_set
output: trained model(image_encoder) by stage2
process: 1. get parameters from source domain
		2. build model, then load param form the trained model of stage1
		3. read the imgs and label from support_set
		4. read the val_data and num_query from target domain
		5. get text_features from model
		6. calculate the loss (with two text_features?)
		7. update the model
		8. test: trained model and val_data & num_query
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train stage2 on support set")
    parser.add_argument(
        "--config_file", default="configs/person/support_set.yml", help="path to config file", type=str)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 1. get parameters from source domain
    num_classes, cam_num, view_num = get_source_domain_param(cfg)

    # 2. build model, then load param form the trained model of stage1
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    model.load_param(cfg.Support_set.Stage1model)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # 优化器和调度器考虑全部弃用
    # 优化器
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)

    # 学习率调度器
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
                                         cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                         cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    # 3. read the imgs and label from support_set
    num_support_set_augmentation = 1
    support_set_data_for_stage2 = get_support_set_data(cfg, num_support_set_augmentation)

    # 4. read the val_data and num_query from target domain
    val_data, num_query = get_val_data(cfg)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        support_set_data_for_stage2,
        val_data,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )