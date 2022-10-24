import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def dice2(y_pred, y_true, eps=1e-7):
#     intersect = (y_pred * y_true).sum()
#     return 1 - 2 * (intersect + eps) / (y_pred.sum() + y_true.sum() + eps)

    iflat = y_pred.view(-1)
    tflat = y_true.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps))

def bce_dice(y_pred, y_true):
    dice_loss = dice2(y_pred, y_true)
    bce_score = nn.BCELoss()
    bce_loss = bce_score(y_pred, y_true)
    
#     return dice_loss + bce_loss
    return dice_loss

# def reverse_one_hot(image):
    
#     x = np.argmax(image, axis = -1)
    
#     return x

# def colour_code_segmentation(image, label_values):
    
#     colour_codes = np.array(label_values)
#     x = colour_codes[image.astype(int)]

#     return x

