import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader import *


def ConvLayer(in_channels, out_channels, stride, padding, activation_fn=nn.LeakyReLU(0.2)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        activation_fn
    )


def TransConvLayer(in_channels, out_channels, activation_fn=nn.ReLU(), dropout=0):
    model = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        activation_fn
    ]
    if dropout > 0:
        model.append(nn.Dropout(dropout))
    return nn.Sequential(*model)


class Discriminator(nn.Module):
    """Discriminator"""

    def __init__(self, channels_lab, features_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_lab, features_dim, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            ConvLayer(features_dim, features_dim * 2, stride=2, padding=1),
            ConvLayer(features_dim * 2, features_dim * 4, stride=2, padding=1),
            ConvLayer(features_dim * 4, features_dim * 8, stride=2, padding=1),
            nn.Conv2d(features_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        x = self.disc(x)
        x = x.view(x.size()[0], -1)
        return x


class Generator(nn.Module):
    """U-Net like Generator"""

    def __init__(self, channel_l, features_dim):
        super().__init__()
        self.encode1 = ConvLayer(channel_l, features_dim, stride=1, padding=0)
        self.encode2 = ConvLayer(features_dim, features_dim * 2, stride=2, padding=1)
        self.encode3 = ConvLayer(features_dim * 2, features_dim * 4, stride=2, padding=1)
        self.encode4 = ConvLayer(features_dim * 4, features_dim * 8, stride=2, padding=1)
        self.encode5 = ConvLayer(features_dim * 8, features_dim * 8, stride=2, padding=1)

        self.decode1 = TransConvLayer(features_dim * 8, features_dim * 8, dropout=0.5)
        self.decode2 = TransConvLayer(features_dim * 16, features_dim * 4, dropout=0.5)
        self.decode3 = TransConvLayer(features_dim * 8, features_dim * 2)
        self.decode4 = TransConvLayer(features_dim * 4, features_dim)

        self.final = nn.Sequential(
            nn.Conv2d(features_dim * 2, 2, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)

        d1 = torch.cat((self.decode1(e5), e4), 1)
        d2 = torch.cat((self.decode2(d1), e3), 1)
        d3 = torch.cat((self.decode3(d2), e2), 1)
        d4 = torch.cat((self.decode4(d3), e1), 1)

        x = self.final(d4)
        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


def test():
    N, in_channels, H, W = 32, 3, 32, 32
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (32, 1)
    gen = Generator(z_dim, in_channels)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (32, 2, 32, 32)
    print("Success")


if __name__ == '__main__':
    test()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(channel_l=1, features_dim=64).to(device)

    # run through the dataset and display the first image of every batch
    for idx, sample in enumerate(test_loader):
        img_l, real_img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

        # generate images with generator model
        fake_img_ab = generator(img_l).detach()
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

        print('sample {}/{}'.format(idx + 1, len(test_loader) + 1))
        fake_img_lab = fake_img_lab.cpu()
        print(fake_img_lab.shape)
        print(fake_img_lab[0].shape)
        print(toRGB(fake_img_lab[0]).shape)
        plt.imshow(toRGB(fake_img_lab[1]))
        plt.show()
