import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
path_img = './data/image_raw'
anno_list = glob.glob('./data/label/*')

# read_data and concatenation
for i in range(len(anno_list)):
    anno_dat = pd.read_csv(anno_list[i])
    if i == 0:
        annotaion_data = anno_dat
    else :
        annotaion_data = pd.concat((annotaion_data, anno_dat), axis = 0)
annotaion_data.reset_index(drop=True ,inplace=True)
annotaion_data.to_csv('./model/maskrcnn/annotation_data.csv', index=False)
annotaion_data.to_csv('./data/annotation_data.csv', index=False)

## make dataset for train & test data
xx_len = []
yy_len = []

anno_data = annotaion_data
anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
img_list = np.unique(anno_data.filename.values)
for img_name in img_list:
    img_name_base = os.path.splitext(img_name)[0]
    ### image preprocessing
    image_path = os.path.join(path_img, img_name)
    img = cv2.imread(image_path)
    img_size = img.shape
    h, w, c = img_size
    resize_factor = max(w, h) // 2000

    r_h = int(1/resize_factor * h)
    r_w = int(1/resize_factor * w)
    re_img = cv2.resize(img, (r_w, r_h), interpolation = cv2.INTER_CUBIC)

    # Annotation preprocessing
    img_anno = anno_data[anno_data.filename == img_name]
    if not eval(img_anno.region_shape_attributes.values[0]):
        print('aaa')
        continue
    # _mask = np.zeros((r_h, r_w,  len(img_anno))).astype(np.uint8)
    for idx in range(len(img_anno)):
       _anno_dict = eval(img_anno.region_shape_attributes.values[idx])
       # _mask[..., idx] = cv2.ellipse(_mask[..., idx].astype(np.uint8),
       #                               (int(_anno_dict['cx']//resize_factor),
       #                                int(_anno_dict['cy']//resize_factor)),
       #                               (int(_anno_dict['rx']//resize_factor),
       #                                int(_anno_dict['ry']//resize_factor)),
       #                               _anno_dict['theta'],
       #                               0, 360,
       #                               idx+1, -1)
       xx_len.append(int(_anno_dict['rx']//resize_factor))
       yy_len.append(int(_anno_dict['ry']//resize_factor))
    #cv2.imwrite(f'./data/image/{os.path.splitext(img_name)[0]}.jpg', re_img)
    #np.save(f'./data/mask/{img_name_base}.mask.npy', _mask)

np.save('./model/maskrcnn/rx_yr_annotated_len_dat.npy', np.array([xx_len, yy_len]))
plt.hist(np.array(xx_len)*2, bins=50)
plt.show()

for i in ['./data/train/image',
          './data/valid/image',
          './data/train/mask',
          './data/valid/mask']:
    os.makedirs(i, exist_ok=True)

mask_list = np.array(sorted(glob.glob('./data/mask/*')))
image_list = np.array(sorted(glob.glob('./data/image/*')))

for i in range(len(mask_list)):
    if os.path.basename(image_list[i]).split('.')[0] != os.path.basename(mask_list[i]).split('.')[0]:
        print(i)
        print('image and mask idx not match')
        break


np.random.seed(777)
idx = np.arange(len(mask_list))
np.random.shuffle(idx)
thr = int(len(idx)*0.2)

valid_id = idx[:thr]
train_id = idx[thr:]

image_list[21]

mask_train_list = mask_list[train_id]
mask_valid_list = mask_list[valid_id]
image_train_list = image_list[train_id]
image_valid_list = image_list[valid_id]


for i in range(len(mask_train_list)):
    src_path = mask_train_list[i]
    dist_path = src_path.replace('/mask','/train/mask')
    shutil.move(src_path, dist_path)

    src_path = image_train_list[i]
    dist_path = src_path.replace('/image','/train/image')
    shutil.move(src_path, dist_path)


for i in range(len(mask_valid_list)):
    src_path = mask_valid_list[i]
    dist_path = src_path.replace('/mask','/valid/mask')
    shutil.move(src_path, dist_path)

    src_path = image_valid_list[i]
    dist_path = src_path.replace('/image','/valid/image')
    shutil.move(src_path, dist_path)


shutil.rmtree('./data/image')
shutil.rmtree('./data/mask')


###
####
# anno_data = pd.read_csv(anno_list[0])
# anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
# img_list0 = np.unique(anno_data.filename.values)
# anno_data = pd.read_csv(anno_list[2])
# anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
#
# anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
# img_list = np.unique(anno_data.filename.values)
#
# img_name = img_list[3]
# img_name_base = os.path.splitext(img_name)[0]
# ### image preprocessing
# image_path = os.path.join(path_img, img_name)
# img = cv2.imread(image_path)
# img_size = img.shape
# h, w, c = img_size
# resize_factor = max(w, h) // 2000
#
# img_anno = anno_data[anno_data.filename == img_name]
# r_h = int(1 / resize_factor * h)
# r_w = int(1 / resize_factor * w)
# re_img = cv2.resize(img, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
# plt.imshow(re_img)
# plt.show()
# for idx in range(len(img_anno)):
#     _anno_dict = eval(img_anno.region_shape_attributes.values[idx])
#     re_img = cv2.ellipse(re_img.astype(np.uint8),
#                                   (int(_anno_dict['cx'] // resize_factor),
#                                    int(_anno_dict['cy'] // resize_factor)),
#                                   (int(_anno_dict['rx'] // resize_factor),
#                                    int(_anno_dict['ry'] // resize_factor)),
#                                   _anno_dict['theta'],
#                                   0, 360,
#                                   (255, 0, 0), -1)
# plt.imshow(re_img)
# plt.show()
#
#
# img_list1 = np.unique(anno_data.filename.values)
# anno_data = pd.read_csv(anno_list[2])
# anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
# img_list2 = np.unique(anno_data.filename.values)
# np.isin(img_list2,img_list1)