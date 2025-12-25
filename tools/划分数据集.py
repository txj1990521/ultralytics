import os
import random
import shutil

data_dir = r'Z:\伊化煤矿\井筒壁漏水渗水检测'
img_dir = os.path.join(data_dir, 'images')
ldl_dir = os.path.join(data_dir, 'labels')

for fld in ['train', 'val']:
    os.makedirs(os.path.join(img_dir, fld), exist_ok=True)
    os.makedirs(os.path.join(ldl_dir, fld), exist_ok=True)
img_flies = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(img_flies)

split_s = int(0.8 * len(img_flies))
train_t, val_v = img_flies[:split_s], img_flies[split_s:]

for f in train_t + val_v:
    tgt = 'train' if f in train_t else 'val'
    shutil.move(os.path.join(img_dir, f), os.path.join(img_dir, tgt, f))
    shutil.move(os.path.join(ldl_dir, f.rsplit('.', 1)[0] + '.txt'),
                os.path.join(ldl_dir, tgt, f.rsplit('.', 1)[0] + '.txt'))

