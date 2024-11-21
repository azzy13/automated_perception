 # encoding: utf-8
import os
import random
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributed as dist

class CocoDataset():
    def __init__(self):

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (800, 1440)
        self.random_size = (18, 32)
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 50
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.001 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (800, 1440)
        self.test_conf = 0.1
        self.nmsthre = 0.7

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        self.path = "coco"

    def get_train_dataset(self, no_aug=False):
        from .yolox_modules import CarlaDataset, TrainTransform, MosaicDetection

        dataset = CarlaDataset(
            data_dir=os.path.dirname(__file__),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        return torch.utils.data.DataLoader(dataset, batch_size=2)


    def get_val_dataset(self):
        from .yolox_modules import CarlaDataset, ValTransform

        valdataset = CarlaDataset(
            data_dir=os.path.dirname(__file__),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='valid',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        return torch.utils.data.DataLoader(valdataset, batch_size=2)
