# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
from torch.cuda.amp import GradScaler, autocast
# visualization
import matplotlib.pyplot as plt
import wandb
from collections import defaultdict


SAVED_DIR = "/data/ephemeral/home/checkpoints/results_unetpp/"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

LR = 1e-3
RANDOM_SEED = 23
LOSS_WEIGHT = 0.5
IMAGE_SIZE = 1536
# 적절하게 조절
NUM_EPOCHS = 100
VAL_EVERY = 3
ACCUMULATION_STEPS = 1
THR=0.5
model = smp.DeepLabV3Plus(encoder_name="efficientnet-b4",
                         encoder_weights="imagenet",
                         in_channels=3, 
                         classes=29)

# # Inference
state_dict = torch.load(os.path.join(SAVED_DIR, "UNet++_Consine_TMax100_lr1e-3_combined_loss_Res_1536_fold2.pt"))

model.load_state_dict(state_dict)

# 데이터 경로를 입력하세요
IMAGE_ROOT = "/data/ephemeral/home/data/test/DCM"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name

def test(model, data_loader, thr=THR):
    model = model.cuda()
    model.eval()
    cnt=0
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)
            # print(outputs.shape)  # torch.Size([1, 29, 1024, 1024])

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            # print(outputs.shape)  # torch.Size([1, 29, 2048, 2048])
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class




tf = A.Compose([
        A.Resize(IMAGE_SIZE,IMAGE_SIZE),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.GaussianBlur(blur_limit=(1, 3), p=0.5),
            ])

test_dataset = XRayInferenceDataset(transforms=tf)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)


rles, filename_and_class = test(model, test_loader)

preds = []
for rle in rles[:len(CLASSES)]:
    pred = decode_rle_to_mask(rle, height=2048, width=2048)
    preds.append(pred)

preds = np.stack(preds, 0)

# # To CSV
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]
df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})


df.to_csv("/data/ephemeral/home/submissions/UNet++_Consine_TMax100_lr1e-3_combined_loss_Res_1536_fold2.csv", index=False)
