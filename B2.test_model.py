import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from src import utils
from torchvision import transforms as T
from src.tangerine_utils import get_model_instance_segmentation, apply_mask # TangerineDataset

if not os.path.exists('./model/maskrcnn'):
    os.makedirs('./model/maskrcnn')

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
        # img_path = self.imgs[idx] #os.path.join(self.root, "image", self.imgs[idx])
        # mask_path = self.masks[idx] #os.path.join(self.root, "mask", self.masks[idx])
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = np.load(mask_path)
        masks = mask.transpose(2, 0, 1)
        masks_list_raw = [masks[i] for i in range(masks.shape[0])]
        get_dat = False

        num_objs = masks.shape[0]

        # h, w, c = img.shape
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

        # if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        img = self.transform_img(img_raw)
        # img = Image.fromarray(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


transform = {'train': A.Compose([
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
                                                batch_size=1,
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
for batch in train_data_loader:
    img_id = batch[1][0]['image_id'].detach().cpu().numpy().astype(int)[0]
    print(img_id)
    boxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(int)
    masks = batch[1][0]['masks'].cpu().numpy().astype(int)
    img = batch[0][0].to(device)
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

    #plt.figure(figsize=(30, 10))
    target_img = (_img * 255).astype(np.uint8).copy()

    #plt.subplot(1, 3, 1)
    #plt.imshow(target_img)
    #plt.xlabel('Input Image', size=20)

    num_t = len(boxes)
    for i in range(num_t):
        target_img3 = cv2.rectangle(target_img,
                                    [boxes[i, 0], boxes[i, 1]],
                                    [boxes[i, 2], boxes[i, 3]],
                                    (255, 0, 0), 5)
    masks = np.sum(masks, axis=0)
    masks[masks > 0] = 1
    target_img = apply_mask(target_img, masks, (0, 0, 255), 0.4)

    #plt.subplot(1, 3, 2)
    #plt.imshow(target_img)
    #plt.xlabel('Ground Truth Tangerine', size=20)

    #plt.subplot(1, 3, 3)
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

    #plt.imshow(out_img)
    #plt.xlabel('Predicted Tangerine', size=20)
    #plt.savefig(f'./fig/{img_id}')
    #plt.show()
    #plt.close()
    target_num.append(num_t)
    output_num.append(num_o)


count_result = {'output': output_num, 'target': target_num}
np.save('./model/maskrcnn/count_train_result.npy', count_result)

count_result = np.load('./model/maskrcnn/count_train_result.npy', allow_pickle=True).item()
target_num_0 = np.array(count_result['target'])
output_num_0 = np.array(count_result['output'])

target_num = target_num_0[target_num_0<100]
output_num = output_num_0[target_num_0<100]
x = np.arange(100)
y = np.arange(100)
corr = round(np.corrcoef(target_num, output_num)[0][1], 3)
plt.plot(x, y, '--')
plt.scatter(target_num, output_num)
plt.title(f'Ground Truth and Detected Tangerine Per Image \n Corr. {corr}')
plt.xlabel('Ground Truth')
plt.ylabel('Detected Tangerine')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.savefig('./fig/scatter_train_plot.jpg')
plt.show()


len(target_num)