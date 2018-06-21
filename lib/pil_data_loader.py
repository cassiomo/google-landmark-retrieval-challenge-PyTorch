import pdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from torchvision import transforms, utils
from torchvision.transforms import functional as F




def show_landmarks(image,landmark_id,text=False):
    """Show image with landmark ids"""
    #image_transform = transforms.ToTensor()
    plt.imshow(image)
    if text:
        plt.text(500,1000,landmark_id, fontsize=10)
    plt.show()
class LandmarksDataset(Dataset):
    """
    Google Landmarks dataset
    """
    def __init__(self,csv_file,root_dir,transform=None):
        self.landmarks_frame=pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,
                str(self.landmarks_frame.iloc[idx,0])+".jpg")
        #convert jpg to PIL (jpg -> Tensor -> PIL)
        image = Image.open(img_name)
        jpg_to_tensor = transforms.ToTensor()
        tensor_to_pil = transforms.ToPILImage()
        image = tensor_to_pil(jpg_to_tensor(image))
        landmark_id = self.landmarks_frame.iloc[idx,2]
        sample = {"image":image,"landmark_id":landmark_id}

        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample
class ToTensor(object):
    """
    Convert PIL Image and ndarray to tensor to Tensor
    """
    pass
landmarks_dataset = LandmarksDataset(csv_file="sample/train_clean.csv",
    root_dir="sample/train/")
fig = plt.figure()
for i in range(len(landmarks_dataset)):
    sample = landmarks_dataset[i]
    print("Sample Index={}".format(i))
    print("Image Size: {}".format(sample['image'].size),
            "Landmark ID: {}".format(sample["landmark_id"]))
    print(type(sample["image"]))
    print(type(sample["landmark_id"]))
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.axis("off")
    show_landmarks(**sample)
    if i==2:
        break
"""
landmarks_frame = pd.read_csv("sample/train_clean.csv")
n = 65
img_name = str(landmarks_frame.iloc[n,0])+".jpg"
landmarks = landmarks_frame.iloc[n,2]

plt.figure()
show_landmarks(Image.open(os.path.join("sample/train/",img_name)),landmarks)
plt.show()
"""
