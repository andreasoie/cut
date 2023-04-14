import os
from typing import List

import numpy as np

ckptdir = "checkpoints/autoferry_cycle_gan"

resume_epochs = np.linspace(25, 400, 16, dtype=int).tolist()

def find_files(dirpath: str, suffix: str) -> List[str]:
    return sorted([os.path.join(rootdir, f) for rootdir, _, files in os.walk(dirpath) for f in files if f.lower().endswith(f".{suffix}")])

ckpts = find_files(ckptdir, "pth")

assert len(ckpts) > 0, f"No checkpoints found in {ckptdir}"

# Remove all files with the same name as the checkpoint
for ckpt in ckpts:
    for resume_epoch in resume_epochs:
        basename = os.path.basename(ckpt)
        basenumb = int(basename.split("_")[0])
        
        if basenumb % 25 != 0:
            if os.path.exists(ckpt):
                os.remove(ckpt)