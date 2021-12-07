# load packages
import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from src.engine import train_one_epoch, evaluate
from torch.utils.data import Dataset
from src import utils
from torchvision import transforms as T
import albumentations as A
import cv2
import time
#load data




class TangerineDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

        self.transform_img = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                        ])
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        #img = Image.open(img_path).convert("RGB")

        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = np.load(mask_path)

        masks = mask.transpose(2, 0, 1)
        masks_list_raw = [masks[i] for i in range(masks.shape[0])]

        get_dat = False
        while get_dat == False:
            transformed = self.transforms(image=img_raw, masks=masks_list_raw)
            img = transformed['image']
            masks_list = transformed['masks']

            masks2 = np.zeros((len(masks_list), *masks_list[0].shape))
            for i in range(len(masks_list)):
                masks2[i] = masks_list[i]

            masks = masks2[np.sum(masks2.reshape(len(masks_list), -1), axis=1) != 0]
            if not masks.tolist():
                continue

            num_objs = masks.shape[0]

            #h, w, c = img.shape
            masks[masks > 0] = 1
            boxes = []
            # end, start of the
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            # 0 x h or w x 0 is faild
            degenerate_boxes = np.array(boxes)[:, 2:] <= np.array(boxes)[:, :2]
            if degenerate_boxes.any():
                continue
            get_dat = True

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        img = self.transform_img(img)
        #img = Image.fromarray(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# model
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# load a model pre-trained on COCO
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    anchor_sizes = ((32,), (64,), (80,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, rpn_anchor_generator=anchor_generator)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)


    return model


transform = A.Compose([A.RandomCrop(width=400, height=400),
                       A.HorizontalFlip(p=0.5),
                       A.RandomBrightnessContrast(p=0.2),
                       ])

#def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = TangerineDataset('./data', transform)
dataset_test = TangerineDataset('./data', transform)

# split the dataset in train and test set
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn
                                          )

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=utils.collate_fn
                                               )

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=30,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    #evaluate(model, data_loader_test, device=device)
torch.save(model.state_dict(), f'./model/model_{epoch}.pth')
print("That's it!")

model.eval()
model.load_state_dict(torch.load(f'./model/model_99.pth'))

import matplotlib.pyplot as plt
import cv2

for batch in data_loader:


    img = batch[0][0].to(device)
    out = model((img,))
    target_img = img.detach().cpu().numpy()
    target_img = target_img.transpose(1,2,0)

    out_box = out[0]['boxes'].detach().cpu().numpy().astype(int)
    out_score = out[0]['scores'].detach().cpu().numpy()
    for i in range(3):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        target_img[...,i] = target_img[...,i]*std[i] + mean[i]
        target_img[..., i] = (target_img[..., i] - np.min(target_img[..., i]))/(np.max(target_img[..., i]) - np.min(target_img[..., i]))


    plt.imshow(target_img)
    plt.show()
    num_c = np.sum(out_score>0.7)

    target_img2 = (target_img*255).astype(np.uint8).copy()
    for i in range(num_c):
        target_img2 = cv2.rectangle(target_img2, [out_box[i,0],out_box[i,2]], [out_box[i,1],out_box[i,3]], (255, 0, 0), 5)
    plt.imshow(target_img2)
    plt.show()

    break





