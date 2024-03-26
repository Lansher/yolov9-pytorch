import os
import numpy as np
import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


from yolo import YoloBody
from nets.yolo_training import weights_init, YOLOLoss, ModelEMA, get_lr_scheduler, set_optimizer_lr
from utils.utils import get_classes, get_anchors, download_weights, seed_everything, show_config, worker_init_fn
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.callbacks import LossHistory, EvalCallback
from utils.utils_fit import fit_one_epoch





if __name__ == '__main__':

    Cuda = True
    
    seed = 11

    distributed = False

    sync_bn = False

    fp16 = False
    classes_path = 'model_data/voc_classes.txt'

    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # model_path   = 'model_data/yolov7_weights.pth'
    model_path   = ''

    input_shape  = [640, 640]

    phi          = 'l'

    pretrained   = False

    mosaic       = True
    mosaic_prob  = 0.5
    mixup        = True
    mixup_prob   = 0.5
    special_aug_ratio = 0.7

    label_smoothing = 0

    Init_Epoch  = 0
    Freeze_Epoch = 50
    Freeze_Batch_Size = 8

    Unfreeze_Epoch = 300
    Unfreeze_Batch_Size = 4

    Freeze_Train = True

    Init_lr = 1e-2 #0.01
    Min_lr  = Init_lr * 0.01

    optimizer_type = 'SGD'
    momentum = 0.937
    weight_decay = 5e-4

    lr_decay_type = 'cos'

    save_period = 10

    save_dir = 'logs'

    eval_flag = True
    eval_period = 10

    num_workers = 4

    train_annotation_path = '2007_train.txt'
    val_annotation_path   = '2007_val.txt'


    seed_everything(seed)

    #---------------------------------------------------#
    #   Get GPU for training
    #---------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    # Default distributed = False
    if distributed:
        print("Gpu Device Count : ", ngpus_per_node)

    else:
        # 'local_rank = 0' means it used single GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print()
        local_rank = 0
        rank = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #---------------------------------------------------#
    #   Download pretrained weight
    #---------------------------------------------------#
    if pretrained:
        if distributed:
            download_weights(phi)  
        else:
            download_weights(phi)  

    #---------------------------------------------------#
    #   Generate Model
    #       when inference or predict/detect also need to generate model   
    #---------------------------------------------------#
    model = YoloBody(anchors_mask=anchors_mask, num_classes=num_classes, phi=phi, pretrained=pretrained)
    # # Same
    # model = YoloBody(anchors_mask, num_classes, phi, pretrained)
    if not pretrained:
        weights_init(model)
        # 'yolo_head_P3.weight', 'yolo_head_P3.bias'
        
    if model_path != '':
        #---------------------------------------------------#
        #   Load keys for pretrained weight and current weight
        #   * Keys just like: 'yolo_head_P3.weight', 'yolo_head_P3.bias', 'yolo_head_P4.weight' ...
        #   * Keys including weight & bias
        #   
        #---------------------------------------------------#
        if local_rank == 0:
            print('Load weights from {}.'.format(model_path))
        # Returns a dictionary containing references to the whole state of the module.
        model_dict = model.state_dict()
        # 'torch.load' uses Python's unpickling facilities but treats storages, 
        # which underlie tensors, specially. They are first deserialized on 
        # the CPU and are then moved to the device they were saved from.
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        # ??
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict, strict=False)


        #---------------------------------------------------#
        #   Print Keys do not corresponding
        #---------------------------------------------------#
        if local_rank == 0:
            print('\nSuccessful load keys: ', str(load_key)[:500], '...\nSuccessful load keys number: ', len(load_key))
            print('\Fail to load keys: ', str(no_load_key)[:500], '...\nFail to load keys number: ', len(no_load_key))
    # Run __init__ function
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir  = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape)

    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler 
        scaler = GradScaler()
    else: 
        scaler = None
    
    # Start training ??
    model_train = model.train()

    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

    ema = ModelEMA(model_train)


    #---------------------------------------------------#
    #   loading corresponding dataset's txt file
    #---------------------------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val   = len(val_lines)

    if local_rank == 0:
        show_config(input_shape=input_shape, model_path=model_path, Unfreeze_Batch_Size=Unfreeze_Batch_Size,
                    Freeze_Batch_Size=Freeze_Batch_Size, num_workers=num_workers, Init_Epoch=Init_Epoch, 
                    UnFreeze_Epoch = Unfreeze_Epoch, Freeze_Epoch = Freeze_Epoch,Freeze_Train = Freeze_Train,
                    optimizer_type=optimizer_type, Init_lr=Init_lr, Min_lr=Min_lr, save_period=save_period, 
                    save_dir=save_dir, num_train=num_train, num_val=num_val, classes_path=classes_path, 
                    anchors_path=anchors_path, anchors_mask=anchors_mask)


        wanted_step = 5e4 if optimizer_type == 'SGD' else 1.5e4
        total_step = num_train // Unfreeze_Batch_Size * Unfreeze_Epoch
        # print(total_step)
        if total_step <= wanted_step:
            # Training instence at least 667
            raise ValueError('Instance for training at least 667.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_Batch_Size) + 1
    
    if True:
        unfreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_Batch_Size if Freeze_Train else Unfreeze_Batch_Size

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        init_lr_fit  = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit   = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------------------#
        #   Chose optimizer according to 'optimizer_type'
        #---------------------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k,v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        
        optimizer = {
            'SGD' : optim.SGD(pg0, init_lr_fit, momentum=momentum, nesterov=True), 
            'adam': optim.Adam(pg0, init_lr_fit, betas=(momentum, 0.999))

        }[optimizer_type]
        optimizer.add_param_group({'params' : pg1, 'weight_dacay' : weight_decay})
        optimizer.add_param_group({'params' : pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, Unfreeze_Epoch)

        epoch_step_train = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step_train == 0 or epoch_step_val == 0:
            raise ValueError('Instance in datasets is not enough! ')
        if ema:
            ema.updates = epoch_step_train * Init_Epoch

        #---------------------------------------------------#
        #   Set dataset loader 
        #   *Dataset loader is different to Dataloader
        #---------------------------------------------------#
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, 
                                    epoch_length=Unfreeze_Epoch, mosaic=mosaic, mosaic_prob=mosaic_prob, 
                                    mixup=mixup, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset   = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask, 
                                    epoch_length=Unfreeze_Epoch, mosaic=False, mosaic_prob=0, 
                                    mixup=False, mixup_prob=0, train=False, special_aug_ratio=0)
    
        if distributed:
            raise ValueError("havn't setting distributed mode now! ")
        else:
            train_sampler = None
            val_sampler   = None
            shuffle       = None
        # Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        gen     = DataLoader(train_dataset, batch_size, shuffle, sampler=train_sampler, num_workers=num_workers,
                        pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        gen_val = DataLoader(val_dataset, batch_size, shuffle, sampler=val_dataset, num_workers=num_workers,
                        pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        if local_rank == 0: 
            eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes,
                                        val_lines, log_dir, cuda=Cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        #---------------------------------------------------#
        #   Finally Start Training Model !!!!
        #---------------------------------------------------#
        for epoch in range(Init_Epoch, Unfreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_Batch_Size

                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                        epoch_step_train=epoch_step_train, epoch_step_val=epoch_step_val, gen=gen, gen_val=gen_val, Epoch=Unfreeze_Epoch,cuda=Cuda, 
                        fp16=fp16, scaler=scaler, save_period=save_period, save_dir=save_dir, local_rank=local_rank)

            writer.add_scalar("Loss/train", yolo_loss, epoch)
        if local_rank == 0:
            loss_history.writer.close()

        #Call flush() method to make sure that all pending events have been written to disk.
        writer.flush()






