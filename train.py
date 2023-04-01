import os
import time
import warnings
from collections import OrderedDict

import torch
from matplotlib import pyplot as plt

import wandb
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import tensor2im

warnings.filterwarnings("ignore")

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

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    total_iters = 0                # the total number of training iterations
    optimize_time = 0.1

    if opt.wandb:
        wandb.init(project="cut", entity="andreasoie")
        wandb.config.update(opt)
    os.makedirs(f"snapshots/{opt.name}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            optimize_start_time = time.time()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                filename = os.path.join("snapshots", opt.name, f"{total_iters}.png")
                save_snapshot_image(model.get_current_visuals(), filename)
                if opt.wandb:
                    wandb.log({"example": wandb.Image(filename)})                

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                meta = {"epoch": epoch, "epoch_iter": epoch_iter, "time_compute": optimize_time, "time_load": t_data, **losses}
                if opt.wandb:
                    wandb.log(meta)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                for save_path in model.save_networks(save_suffix):
                    if opt.wandb:
                        wandb.save(save_path)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving models at the end of epoch %d, iters %d' % (epoch, total_iters))
            for save_path in model.save_networks(epoch):
                if opt.wandb:
                    wandb.save(save_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
