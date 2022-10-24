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
import copy
from model.tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103

def main():
    
    cfg = Config()
    #model = Unet(1)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights=None,     
        in_channels=3,                  
        classes=1,
    )


    if os.path.exists(r'saved/best_model_'+model.__class__.__name__+'.pth'):
        model.load_state_dict(torch.load(r'saved/best_model_'+model.__class__.__name__+'.pth', map_location=cfg.device))
    print('Loaded UNet model.')


    best_iou_score = 0.0
    if os.path.exists(r'saved/best_iou.txt'):
        f = open(r'saved/best_iou.txt', 'r')
        best_iou_score = float(f.read())
        f.close()
    
    cfg.set_optimizer(model)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=cfg.loss, 
        metrics=cfg.metrics, 
        optimizer=cfg.optimizer,
        device=cfg.device,
        verbose=cfg.verbose,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=cfg.loss, 
        metrics=cfg.metrics, 
        device=cfg.device,
        verbose=cfg.verbose,
    )

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

    print('Train Size', len(train_dataset))
    print('Val Size', len(val_dataset))
    

    
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train_batch_size,
                            shuffle=cfg.shuffle,
                            num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.val_batch_size,
                            shuffle=False,
                            num_workers=1)

    
    train_logs_list, valid_logs_list = [], []

    for i in range(0, cfg.epochs):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        cfg.lr_scheduler.step()
        valid_logs = valid_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(copy.deepcopy(model.state_dict()), r'saved/best_model_'+model.__class__.__name__+'.pth')
            print('Model saved!')
            
    print('Best IOU Score:', best_iou_score)
    
    f = open(r'saved/best_iou.txt', 'w')
    f.write(str(best_iou_score))
    f.close()

if __name__ == '__main__':
    main()