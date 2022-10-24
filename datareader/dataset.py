import os
import numpy as np
import random
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        augmentation = None,
        preprocessing = None,
        transform = None
    ):
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._augmentation = augmentation
        self._preprocessing = preprocessing
        self._transform = transform

        self._masks = os.listdir(self._mask_dir)
        self._images = os.listdir(self._image_dir)


        ## class values for background and foreground
        self._class_rgb_values = [[0, 0, 0], [255, 255, 255]]


        self._image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

        self._mask_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])




    def _one_hot_encode_mask_(self, mask):
        semantic_map = []
        for colour in self._class_rgb_values:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis = -1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)

        return semantic_map



    def _get_mask_image(
        self, 
        image_path, 
        mask_path
    ):
        
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ## Encoding mask to one hot vector
        ## Depth = Number of classes
        #mask = self._one_hot_encode_mask_(mask)
        
        #image, mask = image.astype('float64'), mask.astype('float64')

        #image = (image / 255).astype(np.float64)

        if self._preprocessing:
            # preprocessing the images
            data = self._preprocessing(image = image, mask = mask)
            image, mask = data['image'], data['mask']


        if self._augmentation:
            # augmenting the images
            data = self._augmentation(image = image, mask = mask)
            image, mask = data['image'], data['mask']
        
        if self._transform:
            image = self._transform(image)
            mask = self._transform(mask)    

    


        return image, mask



    def __len__(self):
        #print('Total Masks', len(self._masks))
        return len(self._mask_dir)


    def __getitem__(self, index):

        ## getting the name of the mask "1.jpg"
        mask_name = self._masks[index]
        ## generating the root path of the mask
        mask_path = os.path.join(self._mask_dir, mask_name)

        ## image name and mask name are same
        ## number of masks < number of images
        ## So accessing images by mask name
        
        ## getting image rppt path by image path
        image_path = os.path.join(self._image_dir, mask_name)
        
        ## reading the image and the mask

        image , mask = self._get_mask_image(
            image_path=image_path,
            mask_path=mask_path
        )


        return image, mask




if __name__ == "__main__":
    import albumentations as album
    def get_training_augmentation():
        train_transform = [    
            album.HorizontalFlip(p=0.5),

            album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            album.PadIfNeeded(min_height=128, min_width=128, always_apply=True, border_mode=0),
            album.RandomCrop(height=128, width=128, always_apply=True),

            album.IAAAdditiveGaussianNoise(p=0.2),
            album.IAAPerspective(p=0.5),
            album.OneOf(
                [
                    album.HorizontalFlip(p=1),                
                    album.VerticalFlip(p=1),

                ],
                p=1.0,
            ),
            # album.OneOf(
            #     [
            #         album.GridDistortion(p=1),
            #         album.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            #     ], p=0.3),

            album.OneOf(
                [
                    album.RandomRotate90(p=1),
                    #album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                ], p=0.75),

            # album.OneOf(
            #     [
            #         album.CLAHE(clip_limit=2),
            #         album.RandomBrightnessContrast(),            
            #     ], p=0.3)
        ]
        return album.Compose(train_transform)
    
    dataset = SegmentationDataset(
        image_dir = r'dataset/train/image',
        mask_dir = r'dataset/train/mask',
        augmentation=get_training_augmentation()
    )

    def visualize(**images):
        n_images = len(images)
        plt.figure(figsize=(20,8))
        for idx, (name, image) in enumerate(images.items()):
            plt.subplot(1, n_images, idx + 1)
            plt.xticks([]); 
            plt.yticks([])
            # get title from the parameter names
            plt.title(name.replace('_',' ').title(), fontsize=20)
            plt.imshow(image, cmap='gray')
        plt.show()
    

    print(len(dataset))
    
    for i, data in enumerate(dataset):
        print(i)


    # for i in range(0, 10):
    #     random_idx = random.randint(0, len(dataset)-1)
    #     image, mask = dataset[random_idx]


    #     visualize(
    #         original_image = image,
    #         ground_truth_mask = mask
    #     )

        #print(mask)