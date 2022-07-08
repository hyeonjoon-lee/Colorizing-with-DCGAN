import os
from networks import Generator
from dataloader import *
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_path = os.path.join('./checkpoints', 'generator_checkpoint_epoch60.pth')
loaded_checkpoint = torch.load(loaded_path)

generator = Generator(channel_l=1, features_dim=64).to(device)
generator.load_state_dict(loaded_checkpoint["model_state"])

with torch.no_grad():
    # run through the dataset and display the real (top 4 rows) & the corresponding fake images (bottom 4 rows)
    for idx, sample in enumerate(test_loader):
        img_l, img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

        # generate images with generator model
        fake_img_ab = generator(img_l).detach()
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

        img_grid = torchvision.utils.make_grid(torch.concat([img_lab, fake_img_lab]))

        plt.imshow(toRGB(img_grid.cpu()))
        plt.show()
