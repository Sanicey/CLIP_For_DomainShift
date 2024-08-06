import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    # 计算平均值
    loss_meter = AverageMeter()
    # 用于自动混合精度（AMP）训练，可以提高训练速度和效率
    scaler = amp.GradScaler()
    # 用于计算监督对比损失，这是一种用于自监督学习的损失函数
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic() # 获取当前的单调时间（从某个固定点开始的时间）
    logger.info("model: {}".format(model))
    # 初始化两个空列表，用于存储图像特征和标签。
    image_features = []
    labels = []

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device) # [256,3,256,128]
            target = vid.to(device) # 256个
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True) # [256,512] 每一个img获取一个512的feature
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N 标签堆叠成一个新的张量，并将其移动到 CUDA 设备上
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device) # 生成一个随机排列的张量，用于随机选择图像
        for i in range(i_ter+1):
            optimizer.zero_grad()

            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list] # 获取当前批次的目标
            image_features = image_features_list[b_list] # 获取当前批次的图像feature

            # 从 model 获得 text_features
            with amp.autocast(enabled=True):
                text_features = model(label = target, get_text = True)

            # 计算图像到文本和文本到图像的损失
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            # 计算梯度
            scaler.scale(loss).backward()

            # 更新模型参数
            scaler.step(optimizer)

            # 更新缩放器
            scaler.update()

            # 更新损失计数器
            loss_meter.update(loss.item(), img.shape[0])

            # 确保所有CUDA操作都已完成
            torch.cuda.synchronize()

            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1), loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            print("Save stage_1 model.state_dict() !")
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    # 计算训练时间
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
