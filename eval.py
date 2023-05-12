import json
import os
import subprocess
import warnings
from copy import deepcopy
from typing import Union

import numpy as np
from munch import Munch
from tqdm import tqdm

import util.util as util
from data import create_dataset
from fid import calculate_fid_given_paths
from models import create_model
from models.base_model import BaseModel
from models.cut_improved import CUTModel
from models.cycle_gan_model import CycleGANModel
from options.test_options import TestOptions
from util.visualizer import save_images

warnings.filterwarnings("ignore")

cut_options = Munch({
    "dataroot": "/home/andreoi/data/autoferry",
    'results_dir': 'outputs',
    "name": "autoferry_cut",
    "gpu_ids": [0],
    "checkpoints_dir": "/home/andreoi/ckpts",
    "model": "cut",
    "input_nc": 3,
    "output_nc": 3,
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
    "batch_size": 8,
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
    "batch_size_val": 9,
    'batch_size': 8,
    'checkpoints_dir': '/home/andreoi/ckpts', # /home/andreoi/ckpts
    'crop_size': 256,
    'dataroot': '/home/andreoi/data/autoferry',
    'dataset_mode': 'unaligned',
    'direction': 'AtoB',
    'display_winsize': 256,
    'easy_label': 'experiment_name',
    'output_nc': 1,
    'input_nc': 3,
    'epoch': 'latest',
    'gpu_ids': [0], #, 1, 2],
    'init_gain': 0.02,
    'init_type': 'xavier',
    'isTrain': False,
    'load_size': 256,
    'max_dataset_size': np.inf,
    'model': 'cut', # cycle_gan,
    'n_layers_D': 3,
    'name': 'autoferry_cut', # autoferry_cycle_gan
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
    'suffix': '',
    'verbose': False,
    'display_id': -1,
 })

def find_epochs(ckptdir: str):
    return sorted(list(set([int(f.split("_")[0]) for f in os.listdir(ckptdir) if f.endswith(".pth")])))

def generate_images(model: Union[CycleGANModel, CUTModel], validloader, savedir) -> float:
    for data in tqdm(validloader, total=len(validloader), desc=f"Evaluating epoch {resume_epoch}", leave=False):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()   # get image results
        img_paths = model.get_image_paths()     # get image paths
        save_images(savedir, visuals, img_paths)

if __name__ == "__main__":
    # Load options
    opt = deepcopy(cut_options)
    
    # Find all epochs
    # resume_epochs = find_epochs(opt.checkpoints_dir + "/" + opt.name) # opt.checkpoints_dir + "/" + opt.name
    
    # 25 to 400 epochs
    resume_epochs = np.linspace(25, 400, 16).astype(int)
    print("Loading checkpoints = ", opt.checkpoints_dir + "/" + opt.name)
    print("Using model = ", opt.model)
    # Setup Unaligned Dataset
    valdataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    valdataloader = valdataset.dataloader
    
    summary = []
    
    # Evaluate
    for resume_epoch in tqdm(resume_epochs, desc=f"Epoch", total=len(resume_epochs)):
        
        # Create empty folder
        if os.path.exists(opt.results_dir):
            os.system(f"rm -rf {opt.results_dir}")
        os.makedirs(opt.results_dir)
    
        opt.epoch = resume_epoch
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        
        # Generate X images
        generate_images(model, valdataloader, opt.results_dir)
        
        # Calculate FID
        paths = [opt.dataroot + "/testB", opt.results_dir]
        score = calculate_fid_given_paths(paths, opt.load_size, opt.batch_size_val)
        summary.append(score)
        
        if os.path.exists(opt.results_dir):
            os.system(f"rm -rf {opt.results_dir}")
            
    print("FID Summary:")
    for epoch, score in zip(resume_epochs, summary):
        print(f"FID [{str(epoch).rjust(3)}]: {score}")