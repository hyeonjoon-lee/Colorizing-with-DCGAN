import os.path
import sys
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks import Generator, Discriminator, initialize_weights
from dataloader import *

# Hyperparameters
LR_GEN = 3e-4  # Initial learning rate for the generator (different from the paper)
LR_DISC = 6e-5  # Initial learning rate for the discriminator (different from the paper)
BATCH_SIZE = 32  # Batch size is also different from the paper
EPOCH = 200
FEATURES = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting up models
generator = Generator(channel_l=1, features_dim=FEATURES).to(device)
discriminator = Discriminator(channels_lab=3, features_dim=FEATURES).to(device)

# Initializing weights
initialize_weights(generator)
initialize_weights(discriminator)

# Optimizers & Scheduler
opt_gen = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=LR_DISC, betas=(0.5, 0.999))
gen_scheduler = optim.lr_scheduler.StepLR(opt_gen, step_size=4, gamma=0.1)
disc_scheduler = optim.lr_scheduler.StepLR(opt_disc, step_size=4, gamma=0.1)

# Losses
l1_loss = torch.nn.L1Loss(reduction='mean')
disc_loss = torch.nn.BCELoss(reduction='mean')

# Tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer = SummaryWriter(f"logs/Baseline")
step = 0

generator.train()
discriminator.train()

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

for epoch in range(EPOCH):

    for batch_idx, sample in enumerate(train_loader):

        img_l = sample[:, 0:1, :, :].float().to(device)
        img_lab = sample.float().to(device)

        fake_img_ab = generator(img_l)
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1).to(device)

        # Targets for calculating loss
        target_ones = torch.ones(img_lab.size(0), 1).to(device)  # N x 1
        target_zeros = torch.zeros(img_lab.size(0), 1).to(device)  # N x 1

        ### Train Discriminator -> max (log(D(y|x)) + log(1 - D(G(0_z|x)|x)))
        discriminator.zero_grad()

        real = discriminator(img_lab)
        fake = discriminator(fake_img_lab.detach())
        loss_real = disc_loss(real, target_ones * 0.9)  # One Sided Label Smoothing
        loss_fake = disc_loss(fake, target_zeros)

        disc_total_loss = loss_real + loss_fake
        disc_total_loss.backward()
        opt_disc.step()

        ### Train Generator -> min (-log(D(G(0_z|x))) + lambda * |G(0_z|x) - y|_1)
        generator.zero_grad()

        loss_adversarial = disc_loss(discriminator(fake_img_lab), target_ones)
        loss_l1 = l1_loss(img_lab[:, 1:, :, :], fake_img_ab)

        gen_total_loss = 0.01 * loss_adversarial + loss_l1
        gen_total_loss.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            with torch.no_grad():
                print(
                    f"Epoch [{epoch}/{EPOCH}] Batch {batch_idx}/{len(train_loader)} \
                    Loss Disc: {disc_total_loss:.4f}, Loss Gen: {gen_total_loss:.4f}"
                )

                img_grid_real = torchvision.utils.make_grid(img_lab)
                img_grid_fake = torchvision.utils.make_grid(fake_img_lab)

                writer_real.add_image("Real", toRGB(img_grid_real.cpu()), global_step=step, dataformats='HWC')
                writer_fake.add_image("Fake", toRGB(img_grid_fake.cpu()), global_step=step, dataformats='HWC')

                writer.add_scalar('Generator Total Loss', gen_total_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Discriminator Total Loss', disc_total_loss, epoch * len(train_loader) + batch_idx)
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
        generator_path = os.path.join('./checkpoints', 'generator_checkpoint_epoch{}.pth'.format(epoch))
        discriminator_path = os.path.join('./checkpoints', 'discriminator_checkpoint_epoch{}.pth'.format(epoch))

        torch.save(generator_checkpoint, generator_path)
        torch.save(discriminator_checkpoint, discriminator_path)
        print("Model Saved at Epoch {}".format(epoch))

    gen_scheduler.step()
    disc_scheduler.step()
