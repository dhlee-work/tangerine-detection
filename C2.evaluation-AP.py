# load packages
import os

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Subset

from src import utils
from src.engine import evaluate
from src.tangerine_utils import TangerineDataset, get_model_instance_segmentation

# load data
if not os.path.exists('./model/maskrcnn'):
    os.makedirs('./model/maskrcnn')

transform = {'train': A.Compose([A.RandomCrop(width=800, height=800),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.2),
                                 ]),
             'valid': A.Compose([#A.RandomCrop(width=800, height=800),
                                 #A.HorizontalFlip(p=0.5),
                                 #A.RandomBrightnessContrast(p=0.2),
                                 ])
             }
# def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
print('load data')
test_dataset = TangerineDataset('./data/valid2ap/', transform['valid'])


test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=1,
                                               collate_fn=utils.collate_fn
                                               )
print('load model')
# get the model using our helper function
print('train model')
num_epochs = 200
model = get_model_instance_segmentation(num_classes)
# move model to the right device
model.to(device)
print('test model')
model.eval()
model.load_state_dict(torch.load(f'./model/maskrcnn/model_200.pth'))

print('start_eval')
evaluate(model, test_data_loader, device)
print("finish")