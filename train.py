from torch.utils.data import DataLoader
from datasets import coco
import lightning as L
import lightning_main as main

base_dataset = coco.CocoDataset()
train_dataset_loader = base_dataset.get_train_dataset()
val_dataset_loader = base_dataset.get_val_dataset()


'''
model = main.Tracker(
    model_selection="yolox",
    model_params={
        "training": True
    },
    losses={
        "yolox": {}, 
        "yolox_cls": {}, 
        "yolox_obj": {}, 
        "yolox_iou": {}, 
        "yolox_l1": {}
    },
    loss_training="yolox",
    hyperparams={"lr": 0.001},
)
'''

use_l1 = True
num_classes = 1

model = main.Tracker(
    model_selection="yolox",
    model_params={
        "training": True,
        "use_l1": use_l1,
        "num_classes": num_classes,
    },
    losses={
        "yolox": {
            "use_l1": use_l1,
            "num_classes": num_classes,
        },
        "yolox": {
            "use_l1": use_l1,
            "num_classes": num_classes,
        },
        "yolox_cls": {
            "num_classes": num_classes,
        },
        "yolox_obj": {
            "num_classes": num_classes,
        },
        "yolox_iou": {
            "num_classes": num_classes,
        },
    },
    loss_training="yolox",
    hyperparams={"lr": 0.001},
)

trainer = L.Trainer(accelerator="gpu", devices=1, strategy="ddp", num_nodes=1)
trainer.fit(model=model, train_dataloaders=train_dataset_loader, val_dataloaders=val_dataset_loader)