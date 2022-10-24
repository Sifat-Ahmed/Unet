import os
import config

cfg = config.Config()

images = cfg.train_image_dir
masks = cfg.train_mask_dir

for i in os.listdir(masks):
    if i not in os.listdir(images):
        print(i)