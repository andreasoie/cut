import os
import subprocess

import numpy as np
from tqdm import tqdm

import util.util as util
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util.visualizer import save_images

if __name__ == '__main__':
    
    opt = TestOptions().parse()  # get test options
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataloader = dataset.dataloader
    
    assert len(dataloader) == 6, "The number of dataloader should be 6"
    os.makedirs(opt.results_dir, exist_ok=True)
    
    resume_epochs = np.linspace(25, 400, 16, dtype=int).tolist()
    
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