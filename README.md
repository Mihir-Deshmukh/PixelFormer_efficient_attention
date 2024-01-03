# PixelFormer_efficient_attention
Aims to reduce the inference time while maintaining performance for monocular depth estimation. [Report](https://github.com/Mihir-Deshmukh/PixelFormer_efficient_attention/blob/main/Towards%20Low-Latency%20Monocular%20Depth%20Estimation_Report.pdf)

## Modifications
- Added a PixelFormer_new.py to accommodate the new attention mechanisms used.
- Added SAM_cosine.py to use cosine similarity window attention instead of dot product window attention.
- Replaced the window attention with the below two attention mechanisms for fusing encoder and decoder features with global context (instead of the 7*7 window used in the original work) which improves the baseline model performance.
  - Added SAM_efficient.py which implements [Efficient Attention](https://arxiv.org/abs/1812.01243) inside the skip attention module.
  - Added SAM_fast.py which implements [FAVOR+ Attention](https://arxiv.org/abs/2009.14794) inside the skip attention module.

Result Comparison:
![Model Comparison](comparison.png)

## Pretrained Models (NYU DepthV2)
- Download all the models from this [Link](https://drive.google.com/drive/folders/1fVyQnh1IAaJc3OVptSXZ0MkysYpbAVTo?usp=drive_link) and place them in the \pretrained folder.
- Environment setup is mentioned in the PixelFormer.ipynb

## How to Run
- By default, the code is configured to train and evaluate the model which utilizes Efficient Attention.
- Changes to the training config can be done in the ```configs/arguments_train_nyu.txt```
### Training 
- Run the following Python file to train the model on the NYU depthV2 dataset.
  ```
  python pixelformer/train.py configs/arguments_train_nyu.txt
  ```
### Evaluation and Testing
- Run the following two commands to run eval and testing. 
  ```
  python pixelformer/eval.py configs/arguments_eval_nyu.txt
  python pixelformer/test.py configs/arguments_test_nyu.txt
  ```

## Acknowledgements
The code utilized in this project has been adapted from the [PixelFormer](https://github.com/ashutosh1807/PixelFormer) repository. For a comprehensive view of the entire architecture, please refer to the original repository.
