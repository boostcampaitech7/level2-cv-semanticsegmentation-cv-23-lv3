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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
# visualization
import matplotlib.pyplot as plt
import wandb

IMAGE_ROOT = "/data/ephemeral/home/SSJ/data/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/SSJ/data/train/outputs_json"

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

BATCH_SIZE = 3
LR = 1e-3
RANDOM_SEED = 23
LOSS_WEIGHT = 0.5
IMAGE_SIZE = 1536
# 적절하게 조절
NUM_EPOCHS = 100
VAL_EVERY = 3
ACCUMULATION_STEPS = 2
SAVED_DIR = "/data/ephemeral/home/SSJ/code/checkpoints"

if not os.path.isdir(SAVED_DIR):
    os.mkdir(SAVED_DIR)

# 프로젝트 설정 및 초기화
os.environ['WANDB_API_KEY'] = 'YOURKEY'
wandb.init(project="Hand Bone Image Segmentation", entity="alsghks1066-inha-university",config={
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "optimizer": "RMSProp",
    "scheduler": "CosineAnnealingLR",
    "dataset": "Human Bones"
})

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

pngs = sorted(pngs)
jsons = sorted(jsons)

pngs = np.array(pngs)
jsons = np.array(jsons)

class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

class AugmentedXRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False, augmentations=None):
        """
        filenames: 이미지 파일명 리스트
        labelnames: 라벨 파일명 리스트
        transforms: Albumentations 변환
        is_train: 학습 여부
        augmentations: 증강 데이터에 적용할 함수 리스트
        """
        self.filenames = filenames
        self.labelnames = labelnames
        self.transforms = transforms
        self.is_train = is_train
        self.augmentations = augmentations if augmentations else []

        # 데이터셋 확장: 원본 + 증강 데이터
        self.data_indices = []
        for i in range(len(filenames)):
            self.data_indices.append((i, "original"))  # 원본 데이터
            for aug in self.augmentations:
                self.data_indices.append((i, aug))  # 증강 데이터

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index, augmentation_type = self.data_indices[idx]

        image_name = self.filenames[index]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path) / 255.0

        label_name = self.labelnames[index]
        label_path = os.path.join(LABEL_ROOT, label_name)
        label_shape = (image.shape[0], image.shape[1], len(CLASSES))
        label = np.zeros(label_shape, dtype=np.uint8)

        # 라벨 파일 읽기
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        for ann in annotations:
            points = np.array(ann["points"])
            class_idx = CLASS2IND[ann["label"]]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            label[..., class_idx] = mask

        # 데이터 증강 적용
        if augmentation_type != "original":
            image = self.apply_augmentation(image, augmentation_type)
            label = self.apply_augmentation(label, augmentation_type)

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            augmented = self.transforms(**inputs)
            image = augmented["image"]
            label = augmented["mask"] if self.is_train else label

        # PyTorch 텐서로 변환
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        label = label.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

    def apply_augmentation(self, data, augmentation_type):
        if augmentation_type == "flip_lr":  # 좌우 반전
            return np.fliplr(data)
        elif augmentation_type == "flip_ud":  # 상하 반전
            return np.flipud(data)
        elif augmentation_type == "rotate_90":  # 90도 회전
            return np.rot90(data, k=1, axes=(0, 1))
        elif augmentation_type == "rotate_270":  # 270도 회전
            return np.rot90(data, k=3, axes=(0, 1))
        return data
augmentations = ["flip_lr", "flip_ud", "rotate_90", "rotate_270"]
# split train-valid
# 한 폴더 안에 한 인물의 양손에 대한 `.png` 파일이 존재하기 때문에
# 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
# 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
groups = [os.path.dirname(fname) for fname in pngs]

# dummy label
ys = [0 for fname in pngs]

# 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
# 5으로 설정하여 GroupKFold를 수행합니다.
gkf = GroupKFold(n_splits=5)

train_filenames = []
train_labelnames = []
valid_filenames = []
valid_labelnames = []
for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
    # 0번을 validation dataset으로 사용합니다.
    if i == 1:
        valid_filenames += list(pngs[y])
        valid_labelnames += list(jsons[y])

    else:
        train_filenames += list(pngs[y])
        train_labelnames += list(jsons[y])


# define colors
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# utility function
# this does not care overlap
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image

#tf = A.Resize(IMAGE_SIZE, IMAGE_SIZE)
tf = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.5),
])
train_dataset = AugmentedXRayDataset(
    filenames=train_filenames,
    labelnames=train_labelnames,
    transforms=tf,
    is_train=True,
    augmentations=augmentations
    )
valid_dataset = AugmentedXRayDataset(
    filenames=valid_filenames,
    labelnames=valid_labelnames,
    transforms=tf,
    is_train=False
    )

image, label = train_dataset[0]

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# Dice 손실 함수 정의
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return 1 - dice.mean()

# BCE와 Dice 손실을 결합한 손실 함수 정의
def bce_dice_loss(pred, target, bce_weight=0.2):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def save_model(model, file_name='Unetpp_best_Consine_TMax100_lr1e-3_epoch100_MP_combined_loss_Res_1536_aug_order_augment_add.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model.state_dict(), output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation(epoch, model, data_loader, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    dices = []
    total_loss = 0
    cnt = 0

    with torch.no_grad():
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()

            with autocast():
                outputs = model(images)
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)

                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

                # Custom loss 사용
                loss = bce_dice_loss(outputs, masks, LOSS_WEIGHT)
                total_loss += loss.item()
                cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())

    # 평균 손실 및 클래스별 평균 Dice coefficient 계산
    avg_loss = total_loss / cnt
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)

    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    # wandb 로그
    wandb.log({
        "Validation Loss": avg_loss,
        "Validation Dice Coef": torch.mean(dices).item(),
        **{f"Class {c} Dice": d.item() for c, d in zip(CLASSES, dices_per_class)}
    })

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice


def train(model, data_loader, val_loader, optimizer, scheduler):
    scaler= GradScaler()

    print(f'Start training..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.
    wandb.watch(model,log='all')
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.

        for step, (images, masks) in enumerate(data_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            # Forward pass with autocast for mixed precision (FP16)
            with autocast():
                # inference
                outputs = model(images)
                # loss 계산
                loss = bce_dice_loss(outputs, masks,LOSS_WEIGHT)
                loss = loss / ACCUMULATION_STEPS
            # Backward pass with scaling for mixed precision
           
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            

            # step 주기에 따른 loss 출력
            if (step + 1) % (25*ACCUMULATION_STEPS)==0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item() * ACCUMULATION_STEPS, 4)}'
                )
            epoch_loss += loss.item() * ACCUMULATION_STEPS 
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        wandb.log({"Epoch Loss": avg_epoch_loss, "Epoch": epoch + 1})
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Avg Epoch Loss: {avg_epoch_loss:.4f}')

        scheduler.step()

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader)
            print(f"Epoch: {epoch + 1}, Validation Dice: {dice:.4f}")
            wandb.log({"validation_dice": dice, "epoch": epoch+1})

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
                wandb.log({"Best Dice": best_dice})


model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

# Optimizer 정의
optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max= 100, eta_min=1e-6)

set_seed()

train(model, train_loader, valid_loader, optimizer, scheduler)

