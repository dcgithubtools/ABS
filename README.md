# ABS
This repository contains the source code and data for the paper "Adaptive Branch Selection for Accelerate Image Super-Resolution" in The Visual Computer.
# Citation
If you have used this code or data in your research, please cite the following papers:

```BibTeX
@article{
  title   = {Adaptive Branch Selection for Accelerate Image Super-Resolution}
  author  = {Cheng, Ding and Zhongqiu, Zhao and Hao, Shen and Xiufeng Liu}
  year    = {2025}}
```


# How to Train ABS
## Environmental dependencies
python 3.7 + PyTorch 1.13 + CUDA11.6

## Data Preparation
### Train
DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
### get Sub-images
Replace the dataset path in the ..data2patch/data_scripts/extract_subimages.py file, and then run the codes as follows:
```code
cd ../data2patch/data_scripts/
python extract_subimages.py
```

### Test
Test2K,Test4K,Test8K:https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH

## First Stage
The First Stage is to train the three branches. Train the basic SR model as Hard-Branch, and run the codes as follows:
```code
cd ../code/
python basicsr/train.py -opt options/train/train_base.yml
```

To train the SR branch with fewer channels, replace the teacher model path in the ..options/train/train_base_d.yml, and run the codes as follows:
```code
python basicsr/train.py -opt options/train/train_base_d.yml
```

## Second-Stage
The Second-Stage is to train the regressor, replace the three branch path in the ..options/train/train_ABS.yml and run the codes as follows:
```code
python basicsr/train.py -opt options/train/train_ABS.yml
```


