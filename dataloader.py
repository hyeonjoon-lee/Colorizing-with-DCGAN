import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def toRGB(img):
    # transpose back
    img = np.transpose(img.numpy(), (1, 2, 0))
    # transform back
    img[:, :, 0] = (img[:, :, 0] + 1) * 50
    img[:, :, 1] = img[:, :, 1] * 127
    img[:, :, 2] = img[:, :, 2] * 127
    # transform to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    # to int8
    img_rgb = (img_rgb * 255.0).astype(np.uint8)
    return img_rgb


def toLAB(dataset):
    np_rgb = dataset.data.copy().astype(np.float32) / 255.0
    np_lab = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in np_rgb])

    np_lab[:, :, :, 0] = np_lab[:, :, :, 0] / 50 - 1
    np_lab[:, :, :, 1] = np_lab[:, :, :, 1] / 127
    np_lab[:, :, :, 2] = np_lab[:, :, :, 2] / 127

    lab = torch.from_numpy(np.transpose(np_lab, (0, 3, 1, 2)))
    return lab


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())


class LABDataset(torch.utils.data.Dataset):
    """Lab dataset"""

    def __init__(self, train=True):
        self.dataset = toLAB(train_dataset if train else test_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]
        return img


train_loader = DataLoader(LABDataset(train=True), batch_size=32, shuffle=True)
test_loader = DataLoader(LABDataset(train=False), batch_size=32, shuffle=False)

if __name__ == '__main__':
    loader = iter(test_loader)
    print(loader.next().shape)
    plt.imshow(toRGB(loader.next()[4]))
    plt.show()
