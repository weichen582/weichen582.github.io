# [DiffMAE](https://arxiv.org/abs/2301.xxxxx)

Official PyTorch implementation of the paper:

> [**Diffusion Models as Masked Autoencoders (DiffMAE)**](http://arxiv.org/abs/2301.xxxxx)<br>
> [Chen Wei](https://weichen582.github.io/), [Karttikeya Mangalam](https://karttikeya.github.io/), [Po-Yao Huang](http://www.cs.cmu.edu/~poyaoh/), [Yanghao Li](https://lyttonhao.github.io/), [Haoqi Fan](https://haoqifan.github.io/), [Hu Xu](https://howardhsu.github.io/), [Huiyu Wang](https://csrhddlam.github.io/), [Cihang Xie](https://cihangxie.github.io/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/) and [Christoph Feichtenhofer](https://feichtenhofer.github.io/) \
> <br>Meta AI, Johns Hopkins University, UC Santa Cruz<br>

<p align="center">
  <img src="http://weichen582.github.io/files/diffmae_teaser.png" width="750px">
</p>

## Generation and visualization

<p align="center">
  <img src="http://weichen582.github.io/files/diffmae_vis.png" width="775px">
</p>

To sample from our pre-trained DiffMAE models:

1. Download the pre-trained model weights.
2. Perform sampling with:
```
python tools/main.py \
  --cfg configs/diffmae/VIT_L_DIFFMAE_VIS.yaml \
  NUM_GPUS 1 \
  TEST.BATCH_SIZE 256 \
  DATA.PATH_TO_DATA_DIR path_to_your_imagenet \
  DIFFMAE.VIS_CHECKPOINT_FILE_PATH path_to_the_model \
  OUTPUT_DIR ./
```

You can edit the [config](../../configs/diffmae/VIT_L_DIFFMAE_VIS.yaml) file to adjust image resolution, masking ratio, sampling strategy (random or center masking), etc.

## Pre-training and fine-tuning

| arch | resolution | in1k acc@1 | pre-trained ckpt | fine-tuned ckpt |
|:---:|:---:|:---:|:---:| :---:|
| ViT-B | 224x224 | 83.9 | [download]() | [download]()
| ViT-L | 224x224 | 85.8 | [download]() | [download]()
| ViT-H | 224x224 | 86.9 | [download]() | [download]()

To fine-tune with our pretrained model:

1. Download the pre-trained model weights.
2. Perform fine-tuning with:
```
python tools/main.py \
  --cfg configs/diffmae/VIT_L_DIFFMAE_FT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_imagenet \
  TRAIN.CHECKPOINT_FILE_PATH pretrained_model_path \
```
Our results are obtained with effective batch size 256 (per node) * 4 (nodes) = 1024.

To pre-train the model:
```
python tools/main.py \
  --cfg configs/diffmae/VIT_L_DIFFMAE_PT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_imagenet \
```

## Acknowledgement
This codebase borrows from [MAE](https://github.com/facebookresearch/mae), [guided-diffusion](https://github.com/openai/guided-diffusion) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). Thank the authors for open-sourcing.

## License
DiffMAE is released under the [Apache 2.0 license](LICENSE).

## Citation
If you find this repository helpful, please consider citing:
```
@article{DiffMAE2023,
  author  = {Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, Christoph Feichtenhofer},
  journal = {arXiv:2301.xxxxx},
  title   = {Diffusion Models as Masked Autoencoders},
  year    = {2023},
}
```
