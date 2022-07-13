import os.path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from network_new import Generator, Discriminator, initialize_weights
from dataloader import toRGB, LABDataset
import argparse

# # Hyperparameters
# LR_GEN = 2e-4  # Initial learning rate for the generator
# LR_DISC = 1e-5  # Initial learning rate for the discriminator
# BATCH_SIZE = 32  # Batch size
# EPOCH = 50


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalization', type=str, default='batch', choices=['batch', 'instance', 'group'], help='type of normalization in the networks')
    parser.add_argument('--use_global', default=True, type=lambda x: (str(x).lower() == 'true'), help='whether to use global features network')
    parser.add_argument('--use_tanh', default=True, type=lambda x: (str(x).lower() == 'true'), help='whether to use tanh in the last layer of the generator')
    parser.add_argument('--lr_gen', type=float, default=2e-4, help='initial lr for generator')
    parser.add_argument('--lr_disc', type=float, default=1e-5, help='initial lr for discriminator')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--lr_gen_step', type=int, default=10, help='frequency for decaying lr gen')
    parser.add_argument('--lr_disc_step', type=int, default=10, help='frequency for decaying lr disc')
    return parser.parse_args()


def train(arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transforms.ToTensor())
    train_loader = DataLoader(LABDataset(train_dataset), batch_size=arg.batch, shuffle=True)


    # Setting up models
    generator = Generator(channel_l=1, features_dim=64, normalization=arg.normalization, use_global=arg.use_global, use_tanh=arg.use_tanh).to(device)
    discriminator = Discriminator(channels_lab=3, features_dim=64, normalization=arg.normalization).to(device)

    # Initializing weights
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    # Optimizers & Scheduler
    opt_gen = optim.Adam(generator.parameters(), lr=arg.lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=arg.lr_disc, betas=(0.5, 0.999))
    gen_scheduler = optim.lr_scheduler.StepLR(opt_gen, step_size=arg.lr_gen_step)
    disc_scheduler = optim.lr_scheduler.StepLR(opt_disc, step_size=arg.lr_disc_step)

    # Losses
    l1_loss = torch.nn.L1Loss()
    disc_loss = torch.nn.BCELoss()
    class_loss = torch.nn.CrossEntropyLoss()

    path = '{}_global'.format(arg.normalization) if arg.use_global else arg.normalization

    # Tensorboard
    writer_real = SummaryWriter("logs/{}/real".format(path))
    writer_fake = SummaryWriter("logs/{}/fake".format(path))
    writer = SummaryWriter("logs/{}/Losses".format(path))
    step = 0

    generator.train()
    discriminator.train()

    checkpoint_path = os.path.join('./checkpoints', path)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in range(1, arg.epoch + 1):
        print(f'===================================== EPOCH {epoch} =====================================')
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        for batch_idx, (sample, target) in enumerate(train_loader):

            img_l = sample[:, 0:1, :, :].float().to(device)
            img_lab = sample.float().to(device)

            # Targets for calculating loss
            target_ones = torch.ones(img_lab.size(0), 1).to(device)  # N x 1
            target_zeros = torch.zeros(img_lab.size(0), 1).to(device)  # N x 1

            ### Train Generator -> min (-log(D(G(0_z|x))) + lambda * |G(0_z|x) - y|_1)
            opt_gen.zero_grad()

            if arg.use_global:
                fake_img_ab, label = generator(img_l)
            else:
                fake_img_ab = generator(img_l)
            fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)

            loss_adversarial = disc_loss(discriminator(fake_img_lab), target_ones)
            loss_l1 = l1_loss(img_lab[:, 1:, :, :], fake_img_ab)

            if arg.use_global:
                loss_class = class_loss(label, target.to(device))
                gen_total_loss = (1 / 100) * loss_adversarial + loss_l1 + (1 / 300) * loss_class
            else:
                gen_total_loss = (1 / 100) * loss_adversarial + loss_l1

            gen_epoch_loss += gen_total_loss
            gen_total_loss.backward()
            opt_gen.step()

            ### Train Discriminator -> max (log(D(y|x)) + log(1 - D(G(0_z|x)|x)))
            opt_disc.zero_grad()

            real = discriminator(img_lab)
            fake = discriminator(fake_img_lab.detach())
            loss_real = disc_loss(real, target_ones * 0.9)  # One Sided Label Smoothing
            loss_fake = disc_loss(fake, target_zeros)

            disc_total_loss = loss_real + loss_fake
            disc_epoch_loss += disc_total_loss
            disc_total_loss.backward()
            opt_disc.step()

            if batch_idx % 300 == 0:
                with torch.no_grad():
                    if arg.use_global:
                        status = f"Epoch [{epoch}/{arg.epoch}] Batch {batch_idx + 1}/{len(train_loader)}\t" \
                                   f"Loss Disc:{disc_total_loss:.4f} (real:{loss_real:.4f} / fake:{loss_fake:.4f}), " \
                                   f"Loss Gen:{gen_total_loss:.4f} (adversarial:{loss_adversarial:.4f} / l1:{loss_l1:.4f} / class:{loss_class:.4f}) "
                    else:
                        status = f"Epoch [{epoch}/{arg.epoch}] Batch {batch_idx + 1}/{len(train_loader)}\t" \
                                   f"Loss Disc:{disc_total_loss:.4f} (real:{loss_real:.4f} / fake:{loss_fake:.4f}), " \
                                   f"Loss Gen:{gen_total_loss:.4f} (adversarial:{loss_adversarial:.4f} / l1:{loss_l1:.4f}) "
                    print(status)

                    img_grid_real = torchvision.utils.make_grid(img_lab)
                    img_grid_fake = torchvision.utils.make_grid(fake_img_lab)

                    writer_real.add_image("Real", toRGB(img_grid_real.cpu()), global_step=step, dataformats='HWC')
                    writer_fake.add_image("Fake", toRGB(img_grid_fake.cpu()), global_step=step, dataformats='HWC')

                    writer.add_scalar('Generator Total Loss', gen_total_loss, (epoch - 1) * len(train_loader) + batch_idx)
                    writer.add_scalar('Discriminator Total Loss', disc_total_loss, (epoch - 1) * len(train_loader) + batch_idx)
                step += 1

        if epoch % 10 == 0:
            generator_checkpoint = {
                "epoch": epoch,
                "model_state": generator.state_dict(),
                "optim_state": opt_gen.state_dict()
            }
            discriminator_checkpoint = {
                "epoch": epoch,
                "model_state": discriminator.state_dict(),
                "optim_state": opt_disc.state_dict()
            }
            generator_path = os.path.join(checkpoint_path, 'generator_epoch{}.pth'.format(epoch))
            discriminator_path = os.path.join(checkpoint_path, 'discriminator_epoch{}.pth'.format(epoch))

            torch.save(generator_checkpoint, generator_path)
            torch.save(discriminator_checkpoint, discriminator_path)

            print("Model Saved at Epoch {}".format(epoch))

        print(f'Total Generator Loss: {gen_epoch_loss / len(train_loader)}\tTotal Discriminator Loss : {disc_epoch_loss / len(train_loader)}')

        gen_scheduler.step()
        disc_scheduler.step()


if __name__ == '__main__':
    args = get_args()
    print(args)
    train(args)
