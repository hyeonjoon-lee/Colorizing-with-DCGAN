# Colorizing-with-DCGAN
Pytorch implementation of "_Image Colorization using Generative Adversarial Networks_."  
We added extra layers to the generator to incorporate global features, inspired by "_Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors
for Automatic Image Colorization with Simultaneous Classification_." Refer to files having names ending with "_global_features."

## Notes
- The code implementations are based on the paper "_Image Colorization using Generative Adversarial Networks (Nazeri et al., 2018)_" and its official code implementation in tensorflow
- The architecture of the generator in _networks_global_features.py_ is also inspired by the paper "_Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification_," specifically regarding the incorporation of global features.
- The PyTorch implementation of spectral normalization is from [Christian Cosgrove's repo](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan). Using spectral normalization in the generator is not yet implemented. 
