import os.path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from networks import Generator
from dataloader import toRGB, LABDataset
import argparse
from PIL import Image
from ignite.metrics import *
from ignite.engine import *


def eval_step(engine, batch):
    return batch


def test(arg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transforms.ToTensor())
    test_loader = DataLoader(LABDataset(test_dataset), batch_size=arg.batch, shuffle=False)

    path = '{}_global'.format(arg.normalization) if arg.use_global else arg.normalization
    print(path)

    loaded_path = os.path.join('./checkpoints', path, 'generator_epoch100.pth')
    loaded_checkpoint = torch.load(loaded_path)

    generator = Generator(channel_l=1, features_dim=64, normalization=arg.normalization if arg.normalization != 'spectral' else 'batch', use_global=arg.use_global).to(device)
    generator.load_state_dict(loaded_checkpoint["model_state"])

    img_path = {
        'fake': os.path.join('./image', path),
        'real': os.path.join('./image', 'real')
    }
    for img in ['fake', 'real']:
        if not os.path.exists(img_path[img]):
            os.makedirs(img_path[img])
        else:
            for file in os.listdir(img_path[img]):
                file_path = os.path.join(img_path[img], file)
                if os.path.isfile(file_path) and ('real' not in file_path or arg.remove_real):
                    print('Deleting file: {}'.format(file_path))
                    os.remove(file_path)

    mae = torch.nn.L1Loss()
    mean_absolute_error = 0

    default_evaluator = Engine(eval_step)

    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, 'ssim')
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')

    peak_signal = 0
    structural_similarity = 0

    generator.eval()
    with torch.no_grad():
        for idx, (sample, _) in enumerate(test_loader):
            img_l, img_lab = sample[:, 0:1, :, :].to(device), sample.to(device)

            fake_img_ab = generator(img_l)[0].detach() if arg.use_global else generator(img_l).detach()
            fake_img_lab = torch.cat([img_l, fake_img_ab], dim=1)

            mean_absolute_error += mae(fake_img_lab, img_lab)
            state = default_evaluator.run([[fake_img_lab, img_lab]])
            peak_signal += state.metrics['psnr']
            structural_similarity += state.metrics['ssim']

            fake_img_lab = Image.fromarray(toRGB(torchvision.utils.make_grid(fake_img_lab).cpu()))
            img_lab = Image.fromarray(toRGB(torchvision.utils.make_grid(img_lab).cpu()))

            fake_img_lab.save(str(os.path.join(img_path['fake'], f'{path}_{idx:04d}.jpeg')))
            if arg.remove_real:
                img_lab.save(str(os.path.join(img_path['real'], f'real{idx:04d}.jpeg')))

    print("Images successfully saved!")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error / len(test_loader):.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {peak_signal / len(test_loader):.4f}")
    print(f"Structural Similarity Index Measure (SSIM): {structural_similarity / len(test_loader):.4f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalization', type=str, default='batch', choices=['batch', 'instance', 'group', 'spectral'],
                        help='type of normalization in the networks')
    parser.add_argument('--use_global', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use global features network')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--remove_real', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to remove saved real images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test(args)
