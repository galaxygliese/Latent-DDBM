#-*- coding:utf-8 -*-

import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Any
from PIL import Image
from glob import glob 
import numpy as np
import os 

def expand2square(pil_img, background_color=None):
    if background_color is None:
        img = np.asarray(pil_img)
        background_color = np.mean(img, axis=tuple(range(img.ndim-1))).astype(np.uint8)
        background_color = tuple(background_color)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class FolderDataset(Dataset):
    def __init__(self, folder_path:str, image_size:int, transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.samples = glob(os.path.join(self.folder_path, '*'))
        self.image_size = image_size
        if transform is None:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img)),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) -> Any:
        path = self.samples[index]
        img = Image.open(path)
        return self.transform(img)

class FolderPairDataset(Dataset):
    def __init__(self, folder_path:str, image_size:int, transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.samples = glob(os.path.join(self.folder_path, '*'))
        self.image_size = image_size
        if transform is None:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img)),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                # T.Lambda(lambda t: (t * 2) - 1),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) -> Any:
        path = self.samples[index]
        img = Image.open(path).convert("RGB")
        w, h = img.size 
        
        target = img.crop((0, 0, w//2, h))
        condition = img.crop((w//2, 0, w, h))
        target_tensor = self.transform(target)
        cond_tensor = self.transform(condition)
        return target_tensor, cond_tensor

if __name__ == '__main__':
    from torchvision.utils import make_grid, save_image
    from IPython.display import display
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import torch
    def imshow(img):
        img = transforms.functional.to_pil_image(img)
        display(img)
    dataset = FolderPairDataset(folder_path='../datasets/maps/train/', image_size=256)
    print("Dataset:", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    imgs = next(iter(data_loader))
    imgs = imgs[1]
    print(imgs.shape, torch.max(imgs), torch.min(imgs))
    imgs = 0.5*(imgs + 1)

    # グリッド上に並べて1枚の画像にする。
    img = make_grid(imgs)
    img = transforms.functional.to_pil_image(img)
    # imshow(img)

    im = Image.fromarray(np.asarray(img))
    im.save("TEST.png")
    # plt.imshow(img)
    # plt.savefig("plot.png")