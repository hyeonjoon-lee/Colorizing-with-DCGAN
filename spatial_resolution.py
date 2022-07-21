import torch
import torch.nn as nn
from networks import Generator
import matplotlib.pyplot as plt
import pytorch_model_summary
img = torch.zeros(1, 1, 32, 32)
img[0, 0, 15, 15] = 1

g = Generator(channel_l=1, features_dim=64, normalization='batch', use_global=True, use_tanh=False)
print(pytorch_model_summary.summary(g, torch.zeros(1, 1, 32, 32), show_input=True,
                                    show_hierarchical=True, show_parent_layers=True))

for name, p in g.named_parameters():
    if 'weight' in name:
        nn.init.constant_(p, 1)
    elif 'bias' in name:
        nn.init.constant_(p, 0)

g.double()
g.eval()
with torch.no_grad():
    img = torch.cat((img.double(), g(img.double())[0]), 1)

plt.imshow(img[0, 1, :, :] / img[0, 1, :, :].max())
plt.show()