import os
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import util.util as util
from data import create_dataset
from data.unaligned_dataset import UnalignedDataset
from models import create_model
from models.cycle_gan_model import CycleGANModel
from options.test_options import TestOptions
from util.util import save_image, tensor2im
from util.visualizer import save_images


@torch.no_grad()
def save_cherry_image(opt: Namespace, model: CycleGANModel, iteration: int) -> None:
    args = deepcopy(opt)
    args.dataroot = "/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/study_cases_cherry_hidden"
    args.direction == "AtoB"
    args.no_flip = True
    args.phase = "test"
    dataloader_val = DataLoader(dataset=UnalignedDataset(args), batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    assert len(dataloader_val) == 6, "Only 6 are expected"
    model.eval()
    
    generated_images = []
    for i, inputs in enumerate(dataloader_val):
        outputs = model.generate_visuals_for_evaluation(inputs, mode="forward")
        outputs["fake_B"] = outputs["fake_B"].squeeze(0)
        generated_images.append(outputs["fake_B"])
    
    img_grid = make_grid(generated_images, nrow=len(generated_images), padding=0, pad_value=1)
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    img_grid = (img_grid + 1) / 2.0
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(img_grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(os.path.join("cherries", f"{iteration}.png"), dpi=300, transparent=True)

def save_snapshot_image(visuals: OrderedDict, filename: str) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=len(visuals), squeeze=False, figsize=(40, 10))
    for i, (label, image) in enumerate(visuals.items()):
        image = tensor2im(image)
        axs[0, i].imshow(image)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(label, fontsize=25)
    plt.savefig(filename)
    plt.tight_layout()
    plt.close()

if __name__ == '__main__':
    
    opt = TestOptions().parse()  # get test options
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataloader = dataset.dataloader
    
    assert len(dataloader) == 6, "The number of dataloader should be 6"
    os.makedirs(opt.results_dir, exist_ok=True)
    os.makedirs("cherries", exist_ok=True)
    
    resume_epochs = [200, 250, 400]
    # resume_epochs = np.linspace(25, 400, 16, dtype=int).tolist()
    
    for resume_epoch in tqdm(resume_epochs, desc="Epoch", total=len(resume_epochs), leave=False):
        opt.which_epoch = resume_epoch
        opt.epoch = resume_epoch
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()
       
        save_cherry_image(opt, model, resume_epoch)
         
        # OUTDIR = os.path.join(opt.results_dir, f"step_{resume_epoch}")
        # os.makedirs(OUTDIR, exist_ok=True)
        
        # for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating epoch {resume_epoch}", leave=False):
        #     model.set_input(data)
        #     model.test()
        #     visuals = model.get_current_visuals()   # get image results
        #     img_paths = model.get_image_paths()     # get image paths
        #     save_images(OUTDIR, visuals, img_paths)
        # del model