from albumentations.augmentations.functional import normalize
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage
from torch.optim.lr_scheduler import StepLR, MultiplicativeLR



class Config:
    
    def __init__(self):
        self.image_height = 128
        self.image_width = 128

        self.epochs = 100
        #self.optimizer = None
        
        self.train_batch_size = 128
        self.val_batch_size = 32

        self.train_image_dir = r'dataset/train/image'
        self.train_mask_dir = r'dataset/train/mask'
        
        self.val_image_dir = r'dataset/val/image'
        self.val_mask_dir = r'dataset/val/mask'
        
        self.shuffle = True
        self.num_workers = 4


        self.classes = ['background', 'foreground']
        self.class_rgb = [[0, 0, 0], [255, 255, 255]]
        self.verbose = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]


        self.transform = transforms.Compose([
                                            #transforms.ToPILImage(),
                                            #transforms.Resize((self.image_height, self.image_width)),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.RandomVerticalFlip(),
                                            #transforms.RandomRotation(90),
                                            #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 0.8)),
                                            transforms.ToTensor(),
                                            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                        ])
        


    def set_optimizer(self, model):
        self.optimizer = torch.optim.RMSprop([ 
            dict(params=model.parameters(), lr=0.00005),
        ])
        self.lr_scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: 0.5)
        
        

