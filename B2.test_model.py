import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src import utils
from src.tangerine_utils import TangerineDataset, get_model_instance_segmentation, apply_mask





if not os.path.exists('./model/maskrcnn'):
    os.makedirs('./model/maskrcnn')

transform = {'train': A.Compose([A.RandomCrop(width=800, height=800),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.2),
                                 ]),
             'valid': A.Compose([  # A.CenterCrop(width=800, height=800),
                 # A.HorizontalFlip(p=0.5),
                 # A.RandomBrightnessContrast(p=0.2),
             ])
             }
# def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
print('load data')
train_dataset = TangerineDataset('./data/train/', transform['train'])
test_dataset = TangerineDataset('./data/valid/', transform['valid'])

# split the dataset in train and test set
# define training and validation data loaders
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=4,
                                                shuffle=True,
                                                num_workers=1,
                                                collate_fn=utils.collate_fn
                                                )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               collate_fn=utils.collate_fn
                                               )

print('load model')
# get the model using our helper function
model = get_model_instance_segmentation(num_classes, 200, 0.3)

# move model to the right device
model.to(device)
print('test model')
model.eval()
model.load_state_dict(torch.load(f'./model/maskrcnn/model_200.pth'))
# evaluate(model, test_data_loader, device=device)


target_num = []
output_num = []
for batch in test_data_loader:
    img_id = batch[1][0]['image_id'].detach().cpu().numpy().astype(int)
    boxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(int)
    masks = batch[1][0]['masks'].cpu().numpy().astype(int)
    img = batch[0][0].to(device)
    out = model((img,))
    _img = img.detach().cpu().numpy()
    _img = _img.transpose(1, 2, 0)

    out_box = out[0]['boxes'].detach().cpu().numpy().astype(int)
    out_score = out[0]['scores'].detach().cpu().numpy()
    out_mask = out[0]['masks'].detach().cpu().numpy().astype(int)
    for i in range(3):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _img[..., i] = _img[..., i] * std[i] + mean[i]
        _img[..., i] = (_img[..., i] - np.min(_img[..., i])) / (np.max(_img[..., i]) - np.min(_img[..., i]))

    plt.figure(figsize=(30, 10))
    target_img = (_img * 255).astype(np.uint8).copy()

    plt.subplot(1, 3, 1)
    plt.imshow(target_img)
    plt.xlabel('Input Image', size=20)

    num_t = len(boxes)
    for i in range(num_t):
        target_img3 = cv2.rectangle(target_img,
                                    [boxes[i, 0], boxes[i, 1]],
                                    [boxes[i, 2], boxes[i, 3]],
                                    (255, 0, 0), 5)
    masks = np.sum(masks, axis=0)
    masks[masks > 0] = 1
    target_img = apply_mask(target_img, masks, (0, 0, 255), 0.4)

    plt.subplot(1, 3, 2)
    plt.imshow(target_img)
    plt.xlabel('Ground Truth Tangerine', size=20)

    plt.subplot(1, 3, 3)
    num_o = np.sum(out_score > 0.1)
    out_img = (_img * 255).astype(np.uint8).copy()
    for i in range(num_o):
        out_img = cv2.rectangle(out_img, [out_box[i, 0], out_box[i, 1]],
                                [out_box[i, 2], out_box[i, 3]],
                                (255, 0, 0),
                                5)
    out_mask = np.sum(out_mask[:num_o], axis=0)
    out_mask[out_mask > 0] = 1
    out_img = apply_mask(out_img, masks, (0, 0, 255), 0.4)

    plt.imshow(out_img)
    plt.xlabel('Predicted Tangerine', size=20)
    plt.savefig(f'./fig/{img_id}')
    plt.show()
    plt.close()
    target_num.append(num_t)
    output_num.append(num_o)

count_result = {'output': output_num, 'target': output_num}
np.save('./model/maskrcnn/count_result.npy', count_result)

x = np.arange(100)
y = np.arange(100)
corr = round(np.corrcoef(target_num, output_num)[0][1], 3)
plt.plot(x, y, '--')
plt.scatter(target_num, output_num)
plt.title(f'Ground Truth and Detected Tangerine Plot \n Corr. {corr}')
plt.xlabel('Ground Truth')
plt.ylabel('Detected Tangerine')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.savefig('./fig/scatter_plot.jpg')
plt.show()
