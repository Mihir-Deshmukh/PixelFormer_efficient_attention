# PixelFormer_efficient_attention
Improved monocular depth estimation

## Modifications
- Added a PixelFormer_new.py to accommodate the new attention mechanisms used.
- Added SAM_cosine.py to use cosine similarity window attention instead of dot product window attention.
- Replaced the window attention with the below two attention mechanisms for fusing encoder and decoder features with global context.
  - Added SAM_efficient.py which implements [Efficient Attention](https://arxiv.org/abs/1812.01243) inside the skip attention module.
  - Added SAM_fast.py which implements [FAVOR+ Attention](https://arxiv.org/abs/2009.14794) inside the skip attention module

## Pretrained Models
- Download all the models from this [Link](https://drive.google.com/drive/folders/1fVyQnh1IAaJc3OVptSXZ0MkysYpbAVTo?usp=drive_link) and place them in the pre-trained folder.
- Environment setup and how to run are mentioned in the PixelFormer.ipynb

## Acknowledgements
Most of the code has been adapted from [PixelFormer](https://github.com/ashutosh1807/PixelFormer).
