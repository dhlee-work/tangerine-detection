# load packages
import os

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Subset

from src import utils
from src.engine import train_one_epoch, eval_one_epoch
from src.tangerine_utils import TangerineDataset, get_model_instance_segmentation

# load data
if not os.path.exists('./model/maskrcnn'):
    os.makedirs('./model/maskrcnn')

transform = {'train': A.Compose([A.RandomCrop(width=800, height=800),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.2),
                                 ]),
             'valid': A.Compose([A.RandomCrop(width=800, height=800),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.2),
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
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=1,
                                               collate_fn=utils.collate_fn
                                               )

print('load model')
# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=50,
                                               gamma=0.1)

print('train model')
num_epochs = 200

metric = ['lr', 'loss', 'loss_classifier', 'loss_box_reg',
          'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']

history = {'metric': metric,
           'train': {i: [] for i in range(1, num_epochs + 1)},
           'valid': {i: [] for i in range(1, num_epochs + 1)}}

for epoch in range(1, num_epochs + 1):
    # train for one epoch, printing every 10 iterations
    metric_tr_log = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
    history['train'][epoch] = [metric_tr_log.meters[i].avg for i in metric]

    metric_val_log = eval_one_epoch(model, optimizer, test_data_loader, device, epoch, print_freq=100)
    history['valid'][epoch] = [metric_val_log.meters[i].avg for i in metric]

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'./model/maskrcnn/model_{epoch}.pth')
        np.save('./model/maskrcnn/metric_log.npy', history)
torch.save(model.state_dict(), f'./model/model_{epoch}.pth')
np.save('./model/maskrcnn/metric_log.npy', history)
print("finish")
