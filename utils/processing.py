import torch

def img_to_label(target_img, pixel_to_label_dict):
    pixels = list(pixel_to_label_dict.keys())
    output = target_img
    for pixel in pixels:
        output = torch.where(target_img==int(pixel), pixel_to_label_dict[pixel], target_img)
    return output