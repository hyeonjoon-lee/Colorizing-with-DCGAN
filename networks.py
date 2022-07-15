import torch.nn.functional as F
import torch.nn as nn
from dataloader import *
import pytorch_model_summary


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=4, stride=2, padding=1, activation_fn=nn.LeakyReLU(0.2),
                 norm=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = {
            'batch': nn.BatchNorm2d(out_channel),
            'instance': nn.InstanceNorm2d(out_channel),
            'group': nn.GroupNorm(int(out_channel / 4), out_channel) if not out_channel % 4 else nn.GroupNorm(1,
                                                                                                              out_channel)
        }[norm] if norm is not None else norm
        self.activation = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            try:
                x = self.norm(x)
            except:
                pass
        return self.activation(x) if self.activation is not None else x


class FusionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inner_size):
        super().__init__()
        self.conv = ConvLayer(in_channel, out_channel, kernel=1, stride=1, padding=0)
        self.inner_size = inner_size

    def forward(self, l, g):
        x = torch.cat([l, g.repeat(1, 1, self.inner_size, self.inner_size)], dim=1)
        return self.conv(x)


class TransConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=4, stride=2, padding=1, activation_fn=nn.ReLU(), norm=None,
                 dropout=0):
        super().__init__()
        self.trans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = {
            'batch': nn.BatchNorm2d(out_channel),
            'instance': nn.InstanceNorm2d(out_channel),
            'group': nn.GroupNorm(int(out_channel / 4), out_channel) if not out_channel % 4 else nn.GroupNorm(1,
                                                                                                              out_channel)
        }[norm] if norm is not None else norm
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.activation(self.norm(self.trans(x))) if self.norm is not None else self.activation(self.trans(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator
    No normalization in the first layer
    """

    def __init__(self, channels_lab, features_dim, normalization=None):
        super().__init__()
        self.disc = nn.Sequential(
            ConvLayer(channels_lab, features_dim, stride=1, padding=0),  # 64 x 32 x 32
            ConvLayer(features_dim, features_dim * 2, norm=normalization),  # 128 x 16 x 16
            ConvLayer(features_dim * 2, features_dim * 4, norm=normalization),  # 256 x 8 x 8
            ConvLayer(features_dim * 4, features_dim * 8, norm=normalization),  # 512 x 4 x 4
            ConvLayer(features_dim * 8, 1, norm=normalization, stride=1, padding=0, activation_fn=nn.Sigmoid()),
            # 1 x 1 x 1
        )

    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        x = self.disc(x)
        x = x.view(x.size()[0], -1)
        return x


class Generator(nn.Module):
    """
    U-Net like Generator
    No normalization in the first & last layers
    """

    def __init__(self, channel_l, features_dim, normalization=None, use_global=True):
        super().__init__()
        self.use_global = use_global

        # low level
        self.encode1 = ConvLayer(channel_l, features_dim, stride=1, padding=0)  # 64 x 32 x 32
        self.encode2 = ConvLayer(features_dim, features_dim * 2, stride=2, padding=1,
                                 norm=normalization)  # 128 x 16 x 16
        self.encode3 = ConvLayer(features_dim * 2, features_dim * 4, stride=2, padding=1,
                                 norm=normalization)  # 256 x 8 x 8
        self.encode4 = ConvLayer(features_dim * 4, features_dim * 8, stride=2, padding=1,
                                 norm=normalization)  # 512 x 4 x 4
        self.encode5 = ConvLayer(features_dim * 8, features_dim * 8, stride=2, padding=1,
                                 norm=normalization)  # 512 x 2 x 2

        # global features
        if use_global:
            self.mid = nn.Sequential(
                ConvLayer(features_dim * 8, features_dim * 8, kernel=3, stride=1, padding=1, norm=normalization),
                # 512 x 2 x 2
                ConvLayer(features_dim * 8, features_dim * 8, kernel=3, stride=1, padding=1, norm=normalization)
                # 512 x 2 x 2
            )
            self.globalfeat = nn.Sequential(
                ConvLayer(features_dim * 8, features_dim * 8, kernel=3, stride=1, padding=1, norm=normalization),
                # 512 x 2 x 2
                ConvLayer(features_dim * 8, features_dim * 16, kernel=2, stride=1, padding=0, norm=normalization),
                # 1024 x 1 x 1
                ConvLayer(features_dim * 16, features_dim * 8, kernel=1, stride=1, padding=0, norm=normalization)
                # 512 x 1 x 1
            )
            self.fusion = FusionLayer(features_dim * 16, features_dim * 8, 2)  # 512 x 2 x 2
            self.classifier = nn.Sequential(
                ConvLayer(features_dim * 8, features_dim * 4, kernel=1, stride=1, padding=0),  # 256 x 1 x 1
                ConvLayer(features_dim * 4, features_dim * 2, kernel=1, stride=1, padding=0),  # 128 x 1 x 1
                nn.Conv2d(features_dim * 2, 10, kernel_size=1, padding=0)  # 10 x 1 x 1
            )

        # colorization level
        self.decode1 = TransConvLayer(features_dim * 8, features_dim * 8, dropout=0.5,
                                      norm=normalization)  # 512 x 4 x 4
        self.decode2 = TransConvLayer(features_dim * 16, features_dim * 4, dropout=0.5,
                                      norm=normalization)  # 256 x 8 x 8
        self.decode3 = TransConvLayer(features_dim * 8, features_dim * 2, norm=normalization)  # 128 x 16 x 16
        self.decode4 = TransConvLayer(features_dim * 4, features_dim, norm=normalization)  # 64 x 32 x 32
        self.decode5 = ConvLayer(features_dim * 2, 2, kernel=1, stride=1, padding=0,
                                 activation_fn=nn.Tanh())  # 2 x 32 x 32

    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)  # 1 x 35 x 35
        e1 = self.encode1(x)  # 64 x 32 x 32
        e2 = self.encode2(e1)  # 128 x 16 x 16
        e3 = self.encode3(e2)  # 256 x 8 x 8
        e4 = self.encode4(e3)  # 512 x 4 x 4
        e5 = self.encode5(e4)  # 512 x 2 x 2

        if self.use_global:
            m = self.mid(e5)  # 512 x 2 x 2
            g = self.globalfeat(e5)  # 512 x 1 x 1
            f = self.fusion(m, g)  # 512 x 2 x 2
            d1 = torch.cat((self.decode1(f), e4), 1)
        else:
            d1 = torch.cat((self.decode1(e5), e4), 1)  # 1024 x 4 x 4
        d2 = torch.cat((self.decode2(d1), e3), 1)  # 512 x 8 x 8
        d3 = torch.cat((self.decode3(d2), e2), 1)  # 256 x 16 x 16
        d4 = torch.cat((self.decode4(d3), e1), 1)  # 128 x 32 x 32

        x = self.decode5(d4)  # 2 x 32 x 32
        return (x, self.classifier(g).view(-1, 10)) if self.use_global else x


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def test():
    N, in_channels, H, W = 32, 3, 32, 32
    l_dim = 1
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 64)
    initialize_weights(disc)
    assert disc(x).shape == (32, 1)
    gen = Generator(l_dim, 64)
    initialize_weights(gen)
    z = torch.randn((N, l_dim, 32, 32))
    assert gen(z)[0].shape == (32, 2, 32, 32)


if __name__ == '__main__':
    USE_GLOBAL = False

    test()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transforms.ToTensor())

    train_loader = DataLoader(LABDataset(train_dataset), batch_size=32, shuffle=True)
    test_loader = DataLoader(LABDataset(test_dataset), batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(channel_l=1, features_dim=64, normalization='batch', use_global=USE_GLOBAL).to(device)
    discriminator = Discriminator(channels_lab=3, features_dim=64, normalization='batch').to(device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    print(pytorch_model_summary.summary(generator, torch.zeros(1, 1, 32, 32).to(device), show_input=True,
                                        show_hierarchical=True, show_parent_layers=True))
    print(pytorch_model_summary.summary(discriminator, torch.zeros(1, 3, 32, 32).to(device), show_input=True,
                                        show_hierarchical=True, show_parent_layers=True))

    with torch.no_grad():
        # run through the dataset and display the first image of every batch
        for idx, sample in enumerate(test_loader):
            img_l, real_img_lab = sample[0][:, 0:1, :, :].to(device), sample[0].to(device)

            # generate images with generator model
            if USE_GLOBAL:
                fake_img_ab = generator(img_l)[0].detach()
            else:
                fake_img_ab = generator(img_l).detach()

            fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

            print('sample {}/{}'.format(idx + 1, len(test_loader) + 1))
            fake_img_lab = fake_img_lab.cpu()

            plt.imshow(toRGB(fake_img_lab[1]))
            plt.show()
