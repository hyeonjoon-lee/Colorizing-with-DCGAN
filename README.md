# Colorizing-with-DCGAN
Pytorch implementation of "_Image Colorization using Generative Adversarial Networks (Nazeri et al., 2018)_."  

Added extra layers in the generator to incorporate global features, inspired by "_Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors
for Automatic Image Colorization with Simultaneous Classification (Iizuka et al., 2016)_."  

Different types of normalization, specifically batch normalization, instance normalization, group normalization, and spectral normalization were implemented to stabilize the learning of GAN and get satisfying results.  

Furthermore, differentiating from the original implementation by Nazeri et al. (2018), we use kernels of size 3.  


## Notes
- The code implementations are based on the paper "_Image Colorization using Generative Adversarial Networks (Nazeri et al., 2018)_" and its [official code implementation](https://github.com/ImagingLab/Colorizing-with-GANs) in tensorflow.
- The architecture of the generator in [networks.py](https://github.com/hyeonjoon-lee/Colorizing-with-DCGAN/blob/main/networks.py) is also inspired by the paper "_Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification_," specifically regarding the incorporation of global features. Refer to [this link](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/).
- The PyTorch implementation of spectral normalization is from [Christian Cosgrove's repo](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan). We use spectral normalization in the discriminator. 
