# ABS
This repository contains the source code and data for the paper "Adaptive Branch Selection for Accelerate Image Super-Resolution" in The Visual Computer.
# Citation
If you have used this code or data in your research, please cite the following papers:

```BibTeX
@article{
  title   = {Adaptive Branch Selection for Accelerate Image Super-Resolution}
  author  = {Cheng, Ding and Zhongqiu, Zhao and Hao, Shen and Xiufeng Liu}
  journal = {The Visual Computer},
  year    = {2025}}
```


# How to Train ABS
## Environmental dependencies
python 3.7 + PyTorch 1.13 + CUDA11.6

# Data Preparation
## Train
DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
## Run
1.cd ..data2patch/data_scripts/
run python extract_subimages.py


## Test
Test2K,Test4K,Test8K:https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH


# Fisrt Stage
# Second-Stage
