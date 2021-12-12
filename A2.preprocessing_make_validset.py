import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from src import utils
from src.tangerine_utils import TangerineDataset, get_model_instance_segmentation, apply_mask

if not os.path.exists('./data/valid2ap'):
    os.makedirs('./data/valid2ap/image')
    os.makedirs('./data/valid2ap/mask')
np.random.seed(777)
transform = {'train': A.Compose([A.RandomCrop(width=800, height=800),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.2),
                                 ]),
             'valid': A.Compose([A.RandomCrop(width=800, height=800),
             ])
             }
# def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
print('load data')
test_dataset = TangerineDataset('./data/valid/', transform['valid'])

# split the dataset in train and test set
# define training and validation data loaders
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               collate_fn=utils.collate_fn
                                               )
for batch in test_data_loader:
    img_id = batch[1][0]['image_id'].detach().cpu().numpy().astype(int)[0]
    boxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(int)
    masks = batch[1][0]['masks'].cpu().numpy().astype(int)
    masks = masks.transpose(1,2,0)
    img = batch[0][0].to(device)
    _img = img.detach().cpu().numpy()
    _img = _img.transpose(1, 2, 0)
    for i in range(3):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _img[..., i] = _img[..., i] * std[i] + mean[i]
        _img[..., i] = (_img[..., i] - np.min(_img[..., i])) / (np.max(_img[..., i]) - np.min(_img[..., i]))
    _img = (_img * 255).astype(np.uint8)
    _img = _img[...,[2,1,0]]
    cv2.imwrite(f'./data/valid2ap/image/{img_id}.jpg', _img)
    np.save(f'./data/valid2ap/mask/{img_id}.npy', masks)