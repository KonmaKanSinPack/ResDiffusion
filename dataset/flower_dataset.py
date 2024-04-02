import os

import PIL.Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import torchvision.utils
from matplotlib import pyplot as plt


class FlowerDataset(data.Dataset):
    def __init__(self, path):
        super(FlowerDataset, self).__init__()
        self.list_dir = os.listdir(path)
        self.transform = T.Compose([
            T.CenterCrop(500),
            T.Resize(32),
            T.ToTensor()
        ])
        self.imgs = []
        for p in self.list_dir:
            abs_path = os.path.join(path, p)
            self.imgs.append(
                self.transform(PIL.Image.open(abs_path))
            )
        print('load data done')

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, item):
        return self.imgs[item]


if __name__ == '__main__':
    dataset = FlowerDataset('/Data/Machine Learning/Cao-ZiHan/diffusion_mnist_demo/data/102flowers/jpg')
    dl = data.DataLoader(dataset, batch_size=16)
    for x in dl:
        print(x)
        x = torchvision.utils.make_grid(x, nrow=4)
        plt.figure(figsize=(8, 8))
        plt.imshow(x.permute(1, 2, 0).numpy())
        plt.show()
        break
