import os
import shutil
import numpy as np

DIR = "/home/andreoi/data/autoferry_objects"

NEWDIR = "/home/andreoi/data/auto/"

DIR_TRAIN_A = NEWDIR+"trainA"
DIR_TEST_A = NEWDIR+"testA"
DIR_TRAIN_B = NEWDIR+"trainB"
DIR_TEST_B = NEWDIR+"testB"

os.makedirs(NEWDIR, exist_ok=True)
os.makedirs(DIR_TRAIN_A, exist_ok=True)
os.makedirs(DIR_TEST_A, exist_ok=True)
os.makedirs(DIR_TRAIN_B, exist_ok=True)
os.makedirs(DIR_TEST_B, exist_ok=True)

def find_images(path: str) -> list:
    imgs = []
    for top, dirs, files in os.walk(path):
        for pics in files:
            if os.path.join(top, pics).endswith(".png"):
                imgs.append(os.path.join(top, pics))
    return imgs


imgs = find_images(DIR)

np.random.shuffle(imgs)

imgs_infra = [img for img in imgs if "infra" in img]
imgs_optical = [img for img in imgs if "optical" in img]

print(f"Found {len(imgs)} images")
print(f"N OPT = {len(imgs_optical)}")
print(f"N INF = {len(imgs_infra)}")

cutoff = int(len(imgs_optical) * 0.8)

trainA, testA = imgs_optical[:cutoff], imgs_optical[cutoff:]
trainB, testB = imgs_infra[:cutoff], imgs_infra[cutoff:]

print(f"trainA = {len(trainA)}")
print(f"testA = {len(testA)}")
print(f"trainB = {len(trainB)}")
print(f"testA = {len(testB)}")

for img in trainA:
    os.system(f"cp {img} {DIR_TRAIN_A}")

for img in testA:
    os.system(f"cp {img} {DIR_TEST_A}")
    
for img in trainB:
    os.system(f"cp {img} {DIR_TRAIN_B}")
    
for img in testB:
    os.system(f"cp {img} {DIR_TEST_B}")

print(f"trainA = {len(find_images(DIR_TRAIN_A))}")
print(f"testA = {len(find_images(DIR_TEST_A))}")
print(f"trainB = {len(find_images(DIR_TRAIN_B))}")
print(f"testA = {len(find_images(DIR_TEST_B))}")