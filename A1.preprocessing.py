import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob

path_img = './data/image_raw'

anno_list = glob.glob('./data/label/*')



for i in range(len(anno_list)):
    anno_data = pd.read_csv(anno_list[i])

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


        ### Annotation preprocessing
        img_anno = anno_data[anno_data.filename == img_name]
        if not eval(img_anno.region_shape_attributes.values[0]):
            print('aaa')
            continue
        _mask = np.zeros((r_h, r_w,  len(img_anno))).astype(np.uint8)

        for idx in range(len(img_anno)):
            _anno_dict = eval(img_anno.region_shape_attributes.values[idx])
            _mask[..., idx] = cv2.ellipse(_mask[..., idx].astype(np.uint8),
                                          (int(_anno_dict['cx']//resize_factor),
                                           int(_anno_dict['cy']//resize_factor)),
                                          (int(_anno_dict['rx']//resize_factor),
                                           int(_anno_dict['ry']//resize_factor)),
                                          _anno_dict['theta'],
                                          0, 360,
                                          idx+1, -1)
        cv2.imwrite(f'./data/image/{img_name}', re_img)
        np.save(f'./data/mask/{img_name_base}.mask.npy', _mask)





###
####
anno_data = pd.read_csv(anno_list[0])
anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
img_list0 = np.unique(anno_data.filename.values)
anno_data = pd.read_csv(anno_list[2])
anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]

anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
img_list = np.unique(anno_data.filename.values)

img_name = img_list[3]
img_name_base = os.path.splitext(img_name)[0]
### image preprocessing
image_path = os.path.join(path_img, img_name)
img = cv2.imread(image_path)
img_size = img.shape
h, w, c = img_size
resize_factor = max(w, h) // 2000

img_anno = anno_data[anno_data.filename == img_name]
r_h = int(1 / resize_factor * h)
r_w = int(1 / resize_factor * w)
re_img = cv2.resize(img, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
plt.imshow(re_img)
plt.show()
for idx in range(len(img_anno)):
    _anno_dict = eval(img_anno.region_shape_attributes.values[idx])
    re_img = cv2.ellipse(re_img.astype(np.uint8),
                                  (int(_anno_dict['cx'] // resize_factor),
                                   int(_anno_dict['cy'] // resize_factor)),
                                  (int(_anno_dict['rx'] // resize_factor),
                                   int(_anno_dict['ry'] // resize_factor)),
                                  _anno_dict['theta'],
                                  0, 360,
                                  (255, 0, 0), -1)
plt.imshow(re_img)
plt.show()


img_list1 = np.unique(anno_data.filename.values)
anno_data = pd.read_csv(anno_list[2])
anno_data = anno_data[['filename', 'region_id', 'region_shape_attributes']]
img_list2 = np.unique(anno_data.filename.values)
np.isin(img_list2,img_list1)