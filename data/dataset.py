import os
from typing import List, Dict
from os.path import join
import numpy as np
import pickle
import torchvision.transforms as tv
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from utils.transfgu_utils import coco_stuff_id_idx_map, coco_stuff_id_idx_map_coarse
from data.transform import ToTensor, ResizeTensor, NormInput


def get_transform(dataset_name: str, size: int, is_train: bool):
    if "cocostuff" in dataset_name:
        dict_transform = {
            'train': tv.Compose([
                ToTensor(),
                ResizeTensor(size=(size, size), img_only=False)
            ]),
            'val': tv.Compose([
                NormInput(),
                ToTensor(),
                ResizeTensor(size=(size, size), img_only=False)
            ])
        }
    else:
        raise ValueError(f"Not support {dataset_name} {is_train} transform.")

    return dict_transform["train"] if is_train else dict_transform["val"]


def UnsegDataset(
        dataset_name: str,
        data_dir: str,
        is_train: bool,
        seed: int,
        cfg: Dict):
    if dataset_name == "cocostuff":
        dataset = Cocostuff(
            split="train2017" if is_train else "val2017",
            data_dir=os.path.join(data_dir, dataset_name + "27"),
            pseudo_dir=cfg["pseudo_dir"],
            pseudo_size=cfg["pseudo_size"],
            n_stuff=cfg["n_stuff"],
            n_thing=cfg["n_thing"],
            transform=get_transform(dataset_name, cfg["img_size"], is_train)
        )

    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    return dataset


class Cocostuff(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: str,
                 pseudo_dir: str,
                 pseudo_size: int,
                 n_thing: int = 80,
                 n_stuff: int = 91,
                 transform=None
                 ):
        self.split = split
        self.data_dir = data_dir
        self.pseudo_dir = pseudo_dir
        self.pseudo_size = pseudo_size
        self.transform = transform
        self.n_thing = n_thing
        self.n_stuff = n_stuff

        self.JPEGPath = f"{self.data_dir}/images/{self.split}"
        self.PNGPath = f"{self.data_dir}/annotations/{self.split}"
        self.annFile = f"{self.data_dir}/annotations/instances_{self.split}.json"
        self.coco = COCO(self.annFile)
        all_ids = self.coco.imgToAnns.keys()

        samples_list_1, samples_list_2 = [], []

        for id in all_ids:

            img_meta = self.coco.loadImgs(id)
            assert len(img_meta) == 1
            H, W = img_meta[0]['height'], img_meta[0]['width']

            if H < W:
                samples_list_1.append(id)
            else:
                samples_list_2.append(id)

        self.samples_list = samples_list_1 + samples_list_2

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        id = self.samples_list[idx]
        img_meta = self.coco.loadImgs(id)
        assert len(img_meta) == 1
        img_meta = img_meta[0]

        if self.pseudo_dir is not None:
            pseudo_thing_path = os.path.join(self.pseudo_dir, img_meta["file_name"].split(".")[0] +
                                             f'_fg_{self.n_thing}_{self.pseudo_size}x{self.pseudo_size}')

            pseudo_stuff_path = os.path.join(self.pseudo_dir, img_meta["file_name"].split(".")[0] +
                                             f'_bg_{self.n_stuff}_{self.pseudo_size}x{self.pseudo_size}')

            assert os.path.exists(pseudo_thing_path) and os.path.exists(pseudo_stuff_path)

        # image
        image = np.array(Image.open(f"{self.JPEGPath}/{img_meta['file_name']}").convert('RGB'))
        label_cat = np.array(Image.open(f"{self.PNGPath}/{img_meta['file_name'].replace('jpg', 'png')}"))

        if self.num_stuff + self.num_things == 171:
            _coco_id_idx_map = np.vectorize(lambda x: coco_stuff_id_idx_map[x])
        elif self.num_stuff + self.num_things == 27:
            _coco_id_idx_map = np.vectorize(lambda x: coco_stuff_id_idx_map_coarse[x])
        else:
            raise NotImplementedError

        label_cat = _coco_id_idx_map(label_cat)
        sample = dict()
        sample['img'] = image
        sample['label'] = label_cat
        sample['meta'] = {'sample_name': img_meta['file_name'].split('.')[0]}
        sample['pseudo_label_things'] = pickle.load(open(pseudo_thing_path, 'rb'))
        sample['pseudo_label_stuff'] = pickle.load(open(pseudo_stuff_path, 'rb'))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
