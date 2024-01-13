import os
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2
import requests
import zipfile
import torch.nn.functional as F

from dataclasses import dataclass
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from matplotlib import gridspec
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import (
    nn,
    optim
)

block_plot = False




################################################################
############ Step 3 - System, training Configurations ##########
################################################################
def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


seed = 7
set_seed(seed)


@dataclass(frozen=True)
class TrainingConfig:
    DEVICE: torch.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS: int = 31
    LATENT_DIM: int = 128
    BATCH_SIZE: int = 16
    LEARNING_RATE_D: float = 0.0002
    LEARNING_RATE_G: float = 0.0002
    CHECKPOINT_DIR: str = os.path.join('model_checkpoint', 'dcgan_flickr_faces')


class DatasetConfig:
    IMG_HEIGHT: int = 64
    IMG_WIDTH: int = 64
    NUM_CHANNELS: int = 3


################################################################
#################### Step 1 - Datset ###########################
################################################################
def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)


def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("dataset_flickr_faces")
            print("Extracted all")
    except:
        print("Invalid file")


download_file(
    'https://www.dropbox.com/s/crg8qvtix3wg6kx/dataset_flickr_faces.zip?dl=1',
    'dataset_flickr_faces.zip'
)

unzip(zip_file='dataset_flickr_faces.zip')




# Tranforms to resize and convert the pixels between -1 and 1.
transform = transforms.Compose([
    transforms.Resize((DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# Prepare the dataset.
dataset_train = datasets.ImageFolder(
    root='dataset_flickr_faces/',
    transform=transform
)

# Visualize a few images.
plt.figure(figsize=(20, 5))

for image in range(TrainingConfig.BATCH_SIZE):

    nrow = 2
    ncol = min(8, int(TrainingConfig.BATCH_SIZE / nrow))

    for i in range(nrow * ncol):
        image = dataset_train[i][0] / 2 + 0.5
        ax = plt.subplot(nrow, ncol, i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")




# The data loader.
train_dataloader = DataLoader(
    dataset=dataset_train,
    batch_size=TrainingConfig.BATCH_SIZE,
    shuffle=True,
    num_workers=4
)







