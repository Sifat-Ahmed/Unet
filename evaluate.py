import os
import albumentations
from albumentations.augmentations import transforms
import cv2

from model.unet import UNet, Unet
from datareader.dataset import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from utils import helper, augmentations
import random
import segmentation_models_pytorch as smp
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imshow_collection

def main():
    cfg = Config()
    #model = Unet(1)
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights=None,     
        in_channels=3,                  
        classes=1,
    )

    test_dataset = SegmentationDataset(image_dir=cfg.val_image_dir,
                                    mask_dir=cfg.val_mask_dir,
                                    #augmentation=augmentations.get_validation_augmentation(),
                                    #preprocessing=augmentations.get_preprocessing(preprocessing_fn=None)
                                    transform = cfg.transform)

    if os.path.exists(r'saved/best_model_'+model.__class__.__name__+'.pth'):
        model.load_state_dict(torch.load(r'saved/best_model_'+model.__class__.__name__+'.pth', map_location=cfg.device))
    print('Loaded UNet model from this run.')

    model = model.to(cfg.device)
    model.eval()

    print("Test size", len(test_dataset))

    for i, data in enumerate(test_dataset):

        #random_idx = random.randint(0, len(test_dataset)-1)
        image, gt_mask = test_dataset[i]
        #image_vis = crop_image(test_dataset_vis[random_idx][0].astype('uint8'))
        #x_tensor = torch.from_numpy(image).to(cfg.device).unsqueeze(0)
        # Predict test image
        pred_mask = model(image.to(cfg.device).unsqueeze(0))
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        #pred_mask = np.transpose(pred_mask,(1,2,0))
        # Get prediction channel corresponding to building
        #pred_building_heatmap = pred_mask[:,:,select_classes.index('building')]
        #pred_mask = helper.colour_code_segmentation(helper.reverse_one_hot(pred_mask), [[0,0,0], [255,255, 255]])
        
        image = image.detach().squeeze().cpu().numpy()
        image = np.transpose(image,(1,2,0))
        pred_mask = pred_mask.astype(np.uint8)

        #print(pred_mask.shape)
        gt_mask = gt_mask.detach().squeeze().cpu().numpy()
        #gt_mask = np.transpose(gt_mask,(1,2,0))
        #gt_mask = helper.colour_code_segmentation(helper.reverse_one_hot(gt_mask), [[0,0,0], [255,255, 255]])
        #gt_mask = gt_mask.astype(np.uint8)
        _, pred_mask = cv2.threshold(pred_mask, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        imshow_collection([image, pred_mask, gt_mask])
        # plt.imshow(pred_mask)
        plt.show()


if __name__ == '__main__':
    main()