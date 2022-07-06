import torch.nn.functional as F
import torch.nn as nn
from dataloader import *

class Discriminator(nn.Module):
    """Discriminator"""

    def __init__(self, channels_lab, features_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_lab, features_dim, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            self._block(features_dim, features_dim * 2, stride=2, padding=1),
            self._block(features_dim * 2, features_dim * 4, stride=2, padding=1),
            self._block(features_dim * 4, features_dim * 8, stride=2, padding=1),
            nn.Conv2d(features_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, stride, padding, activation_fn=nn.LeakyReLU(0.2)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation_fn
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
        self.encode1 = self.conv_block(channel_l, features_dim, stride=1, padding=0)
        self.encode2 = self.conv_block(features_dim, features_dim * 2, stride=2, padding=1)
        self.encode3 = self.conv_block(features_dim * 2, features_dim * 4, stride=2, padding=1)
        self.encode4 = self.conv_block(features_dim * 4, features_dim * 8, stride=2, padding=1)
        self.encode5 = self.conv_block(features_dim * 8, features_dim * 8, stride=2, padding=1)

        self.decode1 = self.trans_conv_block(features_dim * 8, features_dim * 8, dropout=0.5)
        self.decode2 = self.trans_conv_block(features_dim * 16, features_dim * 4, dropout=0.5)
        self.decode3 = self.trans_conv_block(features_dim * 8, features_dim * 2)
        self.decode4 = self.trans_conv_block(features_dim * 4, features_dim)

        self.final = nn.Sequential(
            nn.Conv2d(features_dim * 2, 2, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, stride, padding, activation_fn=nn.LeakyReLU(0.2)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation_fn
        )

    def trans_conv_block(self, in_channels, out_channels, activation_fn=nn.ReLU(), dropout=0):
        model = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            activation_fn
        ]
        if dropout > 0:
            model.append(nn.Dropout(dropout))
        return nn.Sequential(*model)

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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(channel_l=1, features_dim=64).to(device)
    # run through the dataset and display the first image of every batch
    for idx, sample in enumerate(test_loader):

        img_l, real_img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

        # generate images with bn model
        fake_img_ab_bn = generator(img_l).detach()
        fake_img_lab_bn = torch.cat([img_l, fake_img_ab_bn], dim=1)

        print('sample {}/{}'.format(idx + 1, len(test_loader) + 1))
        fake_img_lab_bn = fake_img_lab_bn.cpu()
        print(fake_img_lab_bn.shape)
        imshow(fake_img_lab_bn[0])
