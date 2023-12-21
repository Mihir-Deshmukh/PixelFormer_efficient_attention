# PixelFormer_efficient_attention
Aims to reduce the inference time while maintaining performance for monocular depth estimation. [Report](https://github.com/Mihir-Deshmukh/PixelFormer_efficient_attention/blob/main/Towards%20Low-Latency%20Monocular%20Depth%20Estimation_Report.pdf)

## Modifications
- Added a PixelFormer_new.py to accommodate the new attention mechanisms used.
- Added SAM_cosine.py to use cosine similarity window attention instead of dot product window attention.
- Replaced the window attention with the below two attention mechanisms for fusing encoder and decoder features with global context which improves the baseline model performance.
  - Added SAM_efficient.py which implements [Efficient Attention](https://arxiv.org/abs/1812.01243) inside the skip attention module.
  - Added SAM_fast.py which implements [FAVOR+ Attention](https://arxiv.org/abs/2009.14794) inside the skip attention module

## Pretrained Models (NYU DepthV2)
- Download all the models from this [Link](https://drive.google.com/drive/folders/1fVyQnh1IAaJc3OVptSXZ0MkysYpbAVTo?usp=drive_link) and place them in the \pretrained folder.
- Environment setup and how to run are mentioned in the PixelFormer.ipynb

## Acknowledgements
Most of the code has been adapted from [PixelFormer](https://github.com/ashutosh1807/PixelFormer). Please refer to their repo for the whole architecture.
