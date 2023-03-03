import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def img_to_label(target_img:torch.Tensor, pixel_to_label_dict:dict):
    pixels = list(pixel_to_label_dict.keys())
    output = target_img
    for pixel in pixels:
        output = torch.where(output==int(pixel), pixel_to_label_dict[pixel], output)
    return output.long()

def label_to_onehot(target:torch.Tensor, num_classes:int):
    """onehot encoding for 1 channel labelmap

    Args:
        target (torch.Tensor): shape (N, 1, H, W) have label values
        num_classes (int): the number of classes
    """
    onehot = torch.zeros((target.shape[0], num_classes, target.shape[1], target.shape[2]), dtype=torch.float64)
    for c in range(num_classes):
        onehot[:, c, :, :] = (target==c)
    return onehot

def pred_to_colormap(pred:np.ndarray, colormap:np.ndarray):
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show_pred = colormap[pred_label] # (N, H, W, 3)
    return show_pred

def save_result_img(input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename:str, save_dir:str, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
    N = input.shape[0]
    show_pred = pred_to_colormap(pred, colormap=colormap)
    for i in range(N):
        input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)
        target_img = np.transpose(np.array([target[i]/255]*3), (1, 2, 0)) # (1, H, W) -> (H, W, 3)
        pred_img = show_pred[i] #(H, W, 3)
        cat_img = np.concatenate((input_img, target_img, pred_img), axis=1) # (H, 3W, 3)
        plt.imsave(os.path.join(save_dir, filename[i]), cat_img)