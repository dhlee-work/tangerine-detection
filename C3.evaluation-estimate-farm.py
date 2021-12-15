import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from src import utils
from src.tangerine_utils import get_model_instance_segmentation, apply_mask

# load data
class TangerineDataset(Dataset):
    def __init__(self, root):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image_1212"))))
        self.transform_img = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                        ])

    def __getitem__(self, idx):
        # load images and masks
        # img_path = self.imgs[idx] #os.path.join(self.root, "image", self.imgs[idx])
        # mask_path = self.masks[idx] #os.path.join(self.root, "mask", self.masks[idx])
        img_path = os.path.join(self.root, "image_1212", self.imgs[idx])
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        h, w, c = img_raw.shape
        resize_factor = max(w, h) // 2000
        r_h = int(1 / resize_factor * h)
        r_w = int(1 / resize_factor * w)
        img = cv2.resize(img_raw, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
        # if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        img = self.transform_img(img)
        return img, img_path

    def __len__(self):
        return len(self.imgs)

np.random.seed(777)
# def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
print('load data')
test_dataset = TangerineDataset('./data/test/')

# split the dataset in train and test set
# define training and validation data loaders
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

output_num = []
for batch in test_data_loader:
    img = batch[0][0].to(device)
    img_id = os.path.splitext(os.path.basename(batch[1][0]))[0]
    print(img_id)
    out = model((img,))

    _img = img.detach().cpu().numpy()
    _img = _img.transpose(1, 2, 0)

    out_box = out[0]['boxes'].detach().cpu().numpy().astype(int)
    out_score = out[0]['scores'].detach().cpu().numpy()
    out_mask = out[0]['masks'].detach().cpu().numpy().squeeze()
    for i in range(3):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _img[..., i] = _img[..., i] * std[i] + mean[i]
        _img[..., i] = (_img[..., i] - np.min(_img[..., i])) / (np.max(_img[..., i]) - np.min(_img[..., i]))

    plt.figure(figsize=(20, 10))
    target_img = (_img * 255).astype(np.uint8).copy()

    plt.subplot(1, 2, 1)
    plt.imshow(target_img)
    plt.xlabel('Input Image', size=20)

    plt.subplot(1, 2, 2)
    num_o = np.sum(out_score > 0.1)
    out_img = (_img * 255).astype(np.uint8).copy()
    for i in range(num_o):
        out_img = cv2.rectangle(out_img, [out_box[i, 0], out_box[i, 1]],
                                [out_box[i, 2], out_box[i, 3]],
                                (255, 0, 0),
                                5)
    out_mask = out[0]['masks'].detach().cpu().numpy().squeeze()
    out_mask = np.sum(out_mask[:num_o], axis=0)
    out_mask[out_mask > 0.8] = 1
    out_img = apply_mask(out_img, out_mask, (0, 0, 255), 0.4)

    plt.imshow(out_img)
    plt.xlabel('Predicted Tangerine', size=20)
    plt.savefig(f'./test_output/{img_id}.jpg')
    plt.close()
    output_num.append(num_o)

mean = round(np.mean(output_num),2)
median = round(np.median(output_num),2)
std = round(np.std(output_num),2)
plt.hist(output_num, bins=8)
plt.title(f'Number Of Tangerines Per Test Images \n mean:{mean}, std:{std}')
plt.xlabel('Count')
plt.ylabel('Freq.')
plt.savefig('./fig/test_image_counts.jpg')
plt.show()