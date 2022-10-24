import os, cv2
import albumentations
from albumentations.augmentations import transforms
import numpy as np
from collections import defaultdict
from model.unet import Unet
from datareader.dataset import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from utils import helper, augmentations
import random
import segmentation_models_pytorch as smp
import torch
import copy


def main():
    
    cfg = Config()
    model = Unet(1)

    if os.path.exists(r'saved/best_model.pth'):
        model.load_state_dict(torch.load(r'saved/best_model.pth', map_location=cfg.device))
    print('Loaded UNet model.')

    # best_iou_score = 0.0
    # if os.path.exists(r'saved/best_iou.txt'):
    #     f = open(r'saved/best_iou.txt', 'r')
    #     best_iou_score = float(f.read())
    #     f.close()
    
    model = model.to(cfg.device)
    cfg.set_optimizer(model)

    train_dataset = SegmentationDataset(image_dir=cfg.train_image_dir,
                                        mask_dir=cfg.train_mask_dir,
                                        augmentation=augmentations.get_training_augmentation(),
                                        #preprocessing=augmentations.get_preprocessing(preprocessing_fn=None)
                                        transform = cfg.transform )

    val_dataset = SegmentationDataset(image_dir=cfg.val_image_dir,
                                    mask_dir=cfg.val_mask_dir,
                                    #augmentation=augmentations.get_validation_augmentation(),
                                    #preprocessing=augmentations.get_preprocessing(preprocessing_fn=None)
                                    transform = cfg.transform)

    
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train_batch_size,
                            shuffle=cfg.shuffle,
                            num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.val_batch_size,
                            shuffle=False,
                            num_workers=1)

    
    train_logs_list, valid_logs_list = [], []

    for i in range(1, cfg.epochs+1):
        metrics = defaultdict(list)
        best_loss = np.inf
        
       
        model.train()
        train_loss = 0
        metric = 0        
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {i}")
            
            for i, (image, mask) in enumerate(tepoch):
            
                image = image.to(cfg.device)
                mask = mask.to(cfg.device) 
                
                cfg.optimizer.zero_grad()
                preds = model(image)
                
                loss = helper.bce_dice(preds, mask)
                loss.backward()
                cfg.optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=train_loss / (i + 1), lr=cfg.scheduler.get_lr()[0])
            
            cfg.lr_scheduler.step()
            metrics['loss'].append(train_loss / (i + 1))
          
        # if best_iou_score < valid_logs['iou_score']:
        #     best_iou_score = valid_logs['iou_score']
        #     torch.save(copy.deepcopy(model.state_dict()), 'saved/best_model.pth')
        #     print('Model saved!')
            
    #print('Best IOU Score:', best_iou_score)
    
    # f = open(r'saved/best_iou.txt', 'w')
    # f.write(str(best_iou_score))
    # f.close()

if __name__ == '__main__':
    main()