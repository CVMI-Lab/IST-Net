
# IST-Net: Prior-free Category-level Pose Estimation with Implicit Space Transformation

This is the official implementation of ***IST-Net***. IST-Net is a clean, simple, and prior-free category-level pose estimator. 


**IST-Net: Prior-free Category-level Pose Estimation with Implicit Space Transformation** <br />
[[Paper](https://arxiv.org/abs/2303.13479)] [[Project Page](https://sites.google.com/view/cvmi-ist-net/)] <br />
[Jianhui Liu](https://scholar.google.com/citations?user=n1JW-jYAAAAJ&hl=en), 
[Yukang Chen](https://scholar.google.com/citations?user=6p0ygKUAAAAJ&hl=en), [Xiaoqing Ye](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=zh-CN), [Xiaojuan Qi](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en)<br />

<p align="center"> <img src="docs/main_fig.png" width="100%"> </p>

## Getting startted
#### ***Prepare the environment***

``` shell
conda create -n istnet python=3.6
conda activate istnet
# The code is tested on pytorch1.10 & CUDA11.3, please choose the properate vesion of torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Dependent packages
pip install gorilla-core
pip install gpustat==1.0.0
pip install opencv-python-headless
pip install matplotlib
pip install scipy
```

#### ***Compiling***
```shell
# Clone this repo
git clone https://github.com/CVMI-Lab/IST-Net.git
# Compile pointnet2
cd model/pointnet2
python setup.py install
```

#### ***Prepare the datasets***
For REAL275 and CAMERA25 datasets, please follow the [instruction](https://github.com/JiehongLin/Self-DPDN) in DPDN.


### Training from scartch
```shell
# gpus refers to the ids of gpu. For single gpu, please set it as 0
python train.py --gpus 0,1 --config config/ist_net_default.yaml
```


### Evaluation
```shell
python test.py --config config/ist_net_default.yaml
```

## Experimental results
If you want to get the same results reported in our paper. You can download the  weights below, and modify the test_path in the yaml file. (Eg. the weights are stored in /.../log/test_istnet/epoch_30.pth. Then fill the test_path in yaml file with /.../log/test_istnet/)

|   | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm | 10 degree 10 cm | Pre-trained | 
|---|---|---|---|---|---|---|---|---|
| IST-Net | 82.5 | 76.6 | 47.5 | 53.4 | 72.1 | 80.5 | 82.6 | [Weights](https://drive.google.com/file/d/1g6PmJU_HasyYDvU5ch1_GFCxKiKnkiZO/view?usp=sharing) |


## Citation
If you find this project useful in your research, please consider citing:
```shell
@article{liu2023prior,
  title={Prior-free Category-level Pose Estimation with Implicit Space Transformation},
  author={Liu, Jianhui and Chen, Yukang and Ye, Xiaoqing and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2303.13479},
  year={2023}
}
```

## Acknowledgement
- Our code is developed upon [DPDN](https://github.com/JiehongLin/Self-DPDN).
- The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019). 
