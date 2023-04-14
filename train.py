import os
import random
import time
import warnings
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from data import create_dataset
from data.unaligned_dataset import UnalignedDataset
from models import create_model
from models.cycle_gan_model import CycleGANModel
from options.train_options import TrainOptions

torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

warnings.filterwarnings("ignore")

@torch.no_grad()
def save_cherry_image(opt: Namespace, model: CycleGANModel, filename: str) -> None:
    args = deepcopy(opt)
    args.dataroot = "/home/andreoi/data/study_cases"
    args.direction == "AtoB"
    args.no_flip = True
    args.phase = "test"
    args.isTrain = False
    dataloader_val = DataLoader(dataset=UnalignedDataset(args), batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    assert len(dataloader_val) == 6, "Only 6 are expected"
    model.eval()
    
    generated_images = []
    for inputs in dataloader_val:
        outputs = model.generate_visuals_for_evaluation(inputs, mode="forward")
        outputs["fake_B"] = outputs["fake_B"].squeeze(0)
        generated_images.append(outputs["fake_B"])
    
    img_grid = make_grid(generated_images, nrow=len(generated_images), padding=0, pad_value=1)
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    img_grid = (img_grid * 0.5) + 0.5
    img_grid = np.clip(img_grid, 0, 1)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(img_grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()
    
    args.isTrain = True
    model.to_train()

def save_snapshot_image(visuals: OrderedDict, filename: str) -> None:
    randidx = 0 # 1 out of BS
    generated_images = []
    real_A = visuals["real_A"][randidx]
    real_B = visuals["real_B"][randidx]
    fake_B = visuals["fake_B"][randidx]
    generated_images = [real_A, fake_B, real_B]
    generated_images = [img.repeat(3, 1, 1) if img.shape[0] == 1 else img for img in generated_images]
    generated_images = [img.squeeze(0) for img in generated_images]
    
    img_grid = make_grid(generated_images, nrow=len(generated_images), padding=0, pad_value=1)
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    img_grid = (img_grid * 0.5) + 0.5
    img_grid = np.clip(img_grid, 0, 1)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(img_grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    model = create_model(opt)
    print(f"Training set: {dataset_size} images")
    

    if opt.wandb:
        wandb.init(project="cyc", entity="andreasoie")
        wandb.config.update(opt)

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("cherries", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    iterations = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        dataset.set_epoch(epoch)
        
        current_lr = model.optimizers[0].param_groups[0]['lr']
        desc = f"Epoch {(epoch):3d} / {(opt.n_epochs + opt.n_epochs_decay):3d}, LR = {current_lr:.6f}"
        for i, data in tqdm(enumerate(dataset), total=(dataset_size // opt.batch_size), desc=desc, colour="cyan"):
            iterations += data["A"].size(0)

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)

            model.set_input(data)
            model.optimize_parameters()
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            if iterations % opt.display_freq == 0:
                filename1 = os.path.join("snapshots", f"{iterations}.png")
                filename2 = os.path.join("cherries", f"{iterations}.png")
                save_snapshot_image(model.get_current_visuals(), filename1)
                save_cherry_image(opt=opt, model=model, filename=filename2)
                if opt.wandb:
                    wandb.log({"snapshot": wandb.Image(filename1)})
                    wandb.log({"cherry": wandb.Image(filename2)})

            if iterations % opt.print_freq == 0:
                losses = model.get_current_losses()
                meta = {"epoch": epoch, "iteration": iterations, **losses}
                if opt.wandb:
                    wandb.log(meta)

        if epoch % opt.save_epoch_freq == 0:
            for save_path in model.save_networks(epoch):
                pass # takes up too much space on W&B
        model.update_learning_rate()                    
