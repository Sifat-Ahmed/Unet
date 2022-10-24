import albumentations as album


def get_training_augmentation():
    train_transform = [    
        # album.HorizontalFlip(p=0.5),

        # album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # album.PadIfNeeded(min_height=128, min_width=128, always_apply=True, border_mode=0),
        # album.RandomCrop(height=128, width=128, always_apply=True),

        # album.IAAAdditiveGaussianNoise(p=0.2),
        # album.IAAPerspective(p=0.5),
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


def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=128, min_width=128, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)