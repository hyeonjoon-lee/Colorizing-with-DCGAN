import os
from networks_global_features import Generator
from dataloader_global_features import *

NORMALIZATION = 'batch'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_path = os.path.join('./checkpoints', '{}_global'.format(NORMALIZATION), 'generator_checkpoint_epoch200.pth')
loaded_checkpoint = torch.load(loaded_path)

generator = Generator(channel_l=1, features_dim=64, normalization=NORMALIZATION, use_global=True).to(device)
generator.load_state_dict(loaded_checkpoint["model_state"])

generator.eval()
with torch.no_grad():
    # run through the dataset and display the real (top 4 rows) & the corresponding fake images (bottom 4 rows)
    for idx, (sample, target) in enumerate(test_loader):
        img_l, img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

        # generate images with generator model
        fake_img_ab = generator(img_l)[0].detach()
        fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

        img_grid = torchvision.utils.make_grid(torch.concat([img_lab, fake_img_lab]))

        plt.imshow(toRGB(img_grid.cpu()))
        plt.show()
