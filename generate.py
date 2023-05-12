import os
import subprocess
import warnings

import numpy as np
from munch import Munch
from tqdm import tqdm

import util.util as util
from data import create_dataset
from models import create_model
from util.visualizer import save_images

warnings.filterwarnings("ignore")

def find_epochs(ckptdir: str): return sorted(list(set([int(f.split("_")[0]) for f in os.listdir(ckptdir) if f.endswith(".pth")])))

cut_options = Munch({
    "model": "cut",
    "name": "autoferry_cut",
    'dataroot': '/home/andreoi/data/study_cases',
    "checkpoints_dir": "/home/andreoi/ckpts",
    'results_dir': 'outputs',
    "input_nc": 3,
    "output_nc": 3,
    "gpu_ids": [0],
    "ngf": 64,
    "ndf": 64,
    "lr": 0.0002,
    "beta1": 0.5,
    "n_epochs": 200,
    "n_epochs_decay": 200,
    "beta2": 0.999,
    "netD": "basic",
    "netG": "resnet_9blocks",
    "n_layers_D": 3,
    "normG": "instance",
    "normD": "instance",
    "init_type": "xavier",
    "init_gain": 0.02,
    "no_dropout": True,
    "no_antialias": False,
    "no_antialias_up": False,
    "dataset_mode": "unaligned",
    "direction": "AtoB",
    "serial_batches": False,
    "num_threads": 4,
    "batch_size_val": 9,
    "batch_size": 1,
    "load_size": 256,
    "crop_size": 256,
    "max_dataset_size": np.inf,
    "preprocess": "resize",
    "no_flip": False,
    "display_winsize": 256,
    "random_scale_max": 3.0,
    "epoch": "latest",
    "verbose": False,
    "stylegan2_G_num_downsampling": 1,
    "update_html_freq": 1000,
    "print_freq": 100,
    "no_html": False,
    "save_latest_freq": 5000,
    "save_epoch_freq": 5,
    "evaluation_freq": 5000,
    "save_by_iter": False,
    "continue_train": False,
    "epoch_count": 1,
    "phase": "test",
    "pretrained_name": None,
    "gan_mode": "lsgan",
    "pool_size": 0,
    "lr_policy": "linear",
    "lr_decay_iters": 50,
    "CUT_mode": "CUT",
    "lambda_GAN": 1.0,
    "lambda_NCE": 1.0,
    "nce_idt": True,
    "nce_layers": "0,4,8,12,16",
    "nce_includes_all_negatives_from_minibatch": False,
    "netF": "mlp_sample",
    "netF_nc": 256,
    "nce_T": 0.07,
    "num_patches": 256,
    "flip_equivariance": False,
    "isTrain": False
})

cyclegan_options = Munch({
    'batch_size': 1,
    'checkpoints_dir': '/home/andreoi/ckpts', # /home/andreoi/ckpts
    'crop_size': 256,
    'dataroot': '/home/andreoi/data/study_cases',
    'dataset_mode': 'unaligned',
    'direction': 'AtoB',
    'output_nc': 1,
    'input_nc': 3,
    'eval': False,
    'gpu_ids': [0],
    'init_gain': 0.02,
    'init_type': 'xavier',
    'isTrain': False,
    'load_size': 256,
    'max_dataset_size': np.inf,
    'model': 'cycle_gan',
    'n_layers_D': 3,
    'name': 'autoferry_cycle_gan',
    'ndf': 64,
    'netD': 'basic',
    'netG': 'resnet_9blocks',
    'ngf': 64,
    'no_antialias': False,
    'no_antialias_up': False,
    'no_dropout': True,
    'no_flip': True,
    'normD': 'instance',
    'normG': 'instance',
    'num_test': 50,
    'num_threads': 4,
    'phase': 'test',
    'preprocess': 'resize',
    'random_scale_max': 3.0,
    'results_dir': 'tmp',
    'serial_batches': False,
    'stylegan2_G_num_downsampling': 1,
    'verbose': False,
    'display_id': -1,
 })

if __name__ == '__main__':
    opt = cut_options
    # opt = TestOptions().parse()  # get test options
    # opt.verbose = False
    # opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # opt.isTrain = False
    # opt.phase = "test"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataloader = dataset.dataloader
    
    assert len(dataset) == 6, "The number of dataloader should be 6, but it is {}".format(dataset)
    os.makedirs(opt.results_dir, exist_ok=True)
    
    # def find_epochs(ckptdir: str): return sorted(list(set([int(f.split("_")[0]) for f in os.listdir(ckptdir) if f.endswith(".pth")])))
    # resume_epochs = find_epochs(opt.checkpoints_dir + "/" + opt.name)
    resume_epochs = np.linspace(25, 400, 16).astype(int)
    
    for resume_epoch in tqdm(resume_epochs, desc="Epoch", total=len(resume_epochs), leave=False):
        opt.which_epoch = resume_epoch
        opt.epoch = resume_epoch
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()
        
        OUTDIR = os.path.join(opt.results_dir, f"step_{resume_epoch}")
        os.makedirs(OUTDIR, exist_ok=True)
        
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating epoch {resume_epoch}", leave=False):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()   # get image results
            img_paths = model.get_image_paths()     # get image paths
            save_images(OUTDIR, visuals, img_paths)
        del model