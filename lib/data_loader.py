import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from skimage import io


def show_landmarks(image,landmark_id,text=False):
    """Show image with landmark ids"""
    plt.imshow(image)
    if text:
        plt.text(500,1000,landmark_id, fontsize=10)
    plt.show()

class LandmarkDataset(Dataset):
    """google landmark dataset"""

    def __init__(self,csv_file,root_dir, transform=None):
        """
        csv_file(str) : path of csv file with url, file names and landmark_id
        root_dir(str) : path of images folder
        transform(optional) : transformation of image
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,str(self.landmarks_frame.iloc[idx,0])+'.jpg')
        image = io.imread(img_name)
        landmark_id = self.landmarks_frame.iloc[idx,2]
        sample = {'image':image, 'landmark_id':landmark_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """
    rescale the image to the given size

    args: output_size(tuple or int): Desired size
    """

if __name__ == "__main__":
    landmarks_dataset = LandmarkDataset(csv_file='data/train.csv', root_dir='data/train')
    fig = plt.figure()
    for i in range(len(landmarks_dataset)):
        sample = landmarks_dataset[i]
        print(i, sample['image'].shape, sample['landmark_id'])

        ax = plt.subplot(1,4, i+1)
        plt.tight_layout()
        ax.set_title("Sample #{} - ID: {}".format(i,sample['landmark_id']))
        ax.axis("off")
        show_landmarks(**sample,text=False)

        if i == 3:
            plt.show()
            break
### Scratch Pad
"""
print("Frequency")
print(landmarks_data_tr.nunique())
print(landmarks_data_te.nunique())
print(landmarks_data_tr.landmark_id.value_counts().head())
print(landmarks_data_tr.landmark_id.value_counts().sum())
print(landmarks_data_tr.landmark_id.value_counts().head()/(landmarks_data_tr.landmark_id.value_counts().sum()))
plt.figure()
plt.imshow(io.imread(os.path.join(folder,_path,"train",img_name)))
plt.show()
#dataset path
folder = '/home/pramod/work/google-landmark-retrieval-challenge-PyTorch/'
_path = 'data/'
landmarks_data_tr = pd.read_csv(folder+_path+'/train.csv')
landmarks_data_te = pd.read_csv(folder+_path+'/test.csv')

print(landmarks_data_tr.head())
n = 50
img_name = str(landmarks_data_tr.iloc[n,0])+'.jpg'
landmark_id = landmarks_data_tr.iloc[n,2]

print("Image Name: {}".format(img_name))
print("Image Category: {}".format(landmark_id))
"""
