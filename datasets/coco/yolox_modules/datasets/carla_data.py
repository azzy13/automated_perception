import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from .datasets_wrapper import Dataset

class CarlaDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(
        self,
        data_dir=None,
        json_file="train.json",
        name="train",
        img_size=(608, 1088),
        #img_size=(304, 544),
        preproc=None,
        path="/isis/home/hasana3/ByteTrack/datasets/coco"
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        #print(f"img_id_len: {len(self.ids)}")
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        #print(self.annotations)
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["id"]
        #video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        #print(f"Anno_id{anno_ids}")
        annotations = self.coco.loadAnns(anno_ids)
        #print(f"Annos:{annotations}")
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
                #logger.info("clean_bbox:: {}".format(obj))

        num_objs = len(objs)
        #logger.info("num_clean_bbox:: {}".format(num_objs))

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, file_name)
        #print(f"Img_info: {img_info}")
        #logger.info("IMG INFO: {}".format(img_info))

        del im_ann, annotations
        #res is a matrix, example - index 1-5
        #img_info is width, height, ID, file_name in that order, example - (640, 640, 432, 'file.jpg)
        #Data is loading
        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        #print(self.annotations, "annotations")
        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        #logger.info("IMG FILE: {}, {}, {}".format(self.data_dir, self.name, file_name))
        img = cv2.imread(img_file)
        assert img is not None
        #print("Initial img shape: ", img.shape)
        return img, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)
        initial_size = img.shape

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
