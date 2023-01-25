

# Contrastive Unpaired Translation (CUT)

> Cloned from https://github.com/taesungp/contrastive-unpaired-translation

Website: [Contrastive Learning for Unpaired Image-to-Image Translation](http://taesung.me/ContrastiveUnpairedTranslation/)  
Authors: [Taesung Park](https://taesung.me/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
Institute: UC Berkeley and Adobe Research<br> In ECCV 2020

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).
```bash
pip install -r requirements.txt
```
### CUT and FastCUT Training and Test

- Download the `grumpifycat` dataset (Fig 8 of the paper. Russian Blue -> Grumpy Cats)
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the CUT model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT
```
 or train the FastCUT model
 ```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_FastCUT --CUT_mode FastCUT
```
The checkpoints will be stored at `./checkpoints/grumpycat_*/web`.

- Test the CUT model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --phase train
```

The test results will be saved to a html file here: `./results/grumpifycat/latest_train/index.html`.

### CUT, FastCUT, and CycleGAN

CUT is trained with the identity preservation loss and with `lambda_NCE=1`, while FastCUT is trained without the identity loss but with higher `lambda_NCE=10.0`. Compared to CycleGAN, CUT learns to perform more powerful distribution matching, while FastCUT is designed as a lighter (half the GPU memory, can fit a larger image), and faster (twice faster to train) alternative to CycleGAN. Please refer to the [paper](https://arxiv.org/abs/2007.15651) for more details.

In the above figure, we measure the percentage of pixels belonging to the horse/zebra bodies, using a pre-trained semantic segmentation model. We find a distribution mismatch between sizes of horses and zebras images -- zebras usually appear larger (36.8\% vs. 17.9\%). Our full method CUT has the flexibility to enlarge the horses, as a means of better matching of the training statistics than CycleGAN. FastCUT behaves more conservatively like CycleGAN.

### Apply a pre-trained CUT model and evaluate FID
To run the pretrained models, run the following.

```bash

# Download and unzip the pretrained models. The weights should be located at
# checkpoints/horse2zebra_cut_pretrained/latest_net_G.pth, for example.
wget http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar
tar -xf pretrained_models.tar

# Generate outputs. The dataset paths might need to be adjusted.
# To do this, modify the lines of experiments/pretrained_launcher.py
# [id] corresponds to the respective commands defined in pretrained_launcher.py
# 0 - CUT on Cityscapes
# 1 - FastCUT on Cityscapes
# 2 - CUT on Horse2Zebra
# 3 - FastCUT on Horse2Zebra
# 4 - CUT on Cat2Dog
# 5 - FastCUT on Cat2Dog
python -m experiments pretrained run_test [id]

# Evaluate FID. To do this, first install pytorch-fid of https://github.com/mseitzer/pytorch-fid
# pip install pytorch-fid
# For example, to evaluate horse2zebra FID of CUT,
# python -m pytorch_fid ./datasets/horse2zebra/testB/ results/horse2zebra_cut_pretrained/test_latest/images/fake_B/
# To evaluate Cityscapes FID of FastCUT,
# python -m pytorch_fid ./datasets/cityscapes/valA/ ~/projects/contrastive-unpaired-translation/results/cityscapes_fastcut_pretrained/test_latest/images/fake_B/
# Note that a special dataset needs to be used for the Cityscapes model. Please read below. 
python -m pytorch_fid [path to real test images] [path to generated images]

```

Note: the Cityscapes pretrained model was trained and evaluated on a resized and JPEG-compressed version of the original Cityscapes dataset. To perform evaluation, please download [this](http://efrosgans.eecs.berkeley.edu/CUT/datasets/cityscapes_val_for_CUT.tar) validation set and perform evaluation. 


### SinCUT Single Image Unpaired Training

To train SinCUT (single-image translation, shown in Fig 9, 13 and 14 of the paper), you need to

1. set the `--model` option as `--model sincut`, which invokes the configuration and codes at `./models/sincut_model.py`, and
2. specify the dataset directory of one image in each domain, such as the example dataset included in this repo at `./datasets/single_image_monet_etretat/`. 

For example, to train a model for the [Etretat cliff (first image of Figure 13)](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/imgs/singleimage.gif), please use the following command.

```bash
python train.py --model sincut --name singleimage_monet_etretat --dataroot ./datasets/single_image_monet_etretat
```

or by using the experiment launcher script,
```bash
python -m experiments singleimage run 0
```

For single-image translation, we adopt network architectural components of [StyleGAN2](https://github.com/NVlabs/stylegan2), as well as the pixel identity preservation loss used in [DTN](https://arxiv.org/abs/1611.02200) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py#L160). In particular, we adopted the code of [rosinality](https://github.com/rosinality/stylegan2-pytorch), which exists at `models/stylegan_networks.py`.

The training takes several hours. To generate the final image using the checkpoint,

```bash
python test.py --model sincut --name singleimage_monet_etretat --dataroot ./datasets/single_image_monet_etretat
```

or simply

```bash
python -m experiments singleimage run_test 0
```

### [Datasets](./docs/datasets.md)
Download CUT/CycleGAN/pix2pix datasets. For example,

```bash
bash datasets/download_cut_datasets.sh horse2zebra
```

The Cat2Dog dataset is prepared from the AFHQ dataset. Please visit https://github.com/clovaai/stargan-v2 and download the AFHQ dataset by `bash download.sh afhq-dataset` of the github repo. Then reorganize directories as follows.
```bash
mkdir datasets/cat2dog
ln -s datasets/cat2dog/trainA [path_to_afhq]/train/cat
ln -s datasets/cat2dog/trainB [path_to_afhq]/train/dog
ln -s datasets/cat2dog/testA [path_to_afhq]/test/cat
ln -s datasets/cat2dog/testB [path_to_afhq]/test/dog
```

The Cityscapes dataset can be downloaded from https://cityscapes-dataset.com.
After that, use the script `./datasets/prepare_cityscapes_dataset.py` to prepare the dataset. 


#### Preprocessing of input images

The preprocessing of the input images, such as resizing or random cropping, is controlled by the option `--preprocess`, `--load_size`, and `--crop_size`. The usage follows the [CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo. 

For example, the default setting `--preprocess resize_and_crop --load_size 286 --crop_size 256` resizes the input image to `286x286`, and then makes a random crop of size `256x256` as a way to perform data augmentation. There are other preprocessing options that can be specified, and they are specified in [base_dataset.py](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/data/base_dataset.py#L82). Below are some example options. 

 - `--preprocess none`: does not perform any preprocessing. Note that the image size is still scaled to be a closest multiple of 4, because the convolutional generator cannot maintain the same image size otherwise. 
 - `--preprocess scale_width --load_size 768`: scales the width of the image to be of size 768.
 - `--preprocess scale_shortside_and_crop`: scales the image preserving aspect ratio so that the short side is `load_size`, and then performs random cropping of window size `crop_size`.

More preprocessing options can be added by modifying [`get_transform()`](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/data/base_dataset.py#L82) of `base_dataset.py`. 


### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

If you use the original [pix2pix](https://phillipi.github.io/pix2pix/) and [CycleGAN](https://junyanz.github.io/CycleGAN/) model included in this repo, please cite the following papers
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
