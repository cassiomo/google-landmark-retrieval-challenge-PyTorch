import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Dataloader

from skimage import io

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

def show_landmarks(image,landmark_id):
    """Show image with landmark ids"""
    plt.imshow(image)
    plt.text(1,1,landmark_id, fontsize=20)
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
        return(self.landmarks_frame)
    def __classes__(self):
        return(landmarks_frame.nunique())
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,str(self.landmarks_frame.iloc[idx,0])+'.jpg')
        image = io.imread(img_name)
        landmark_id = self.landmarks_frame.iloc[idx,2]
        sample = {'image':image, 'landmark_id':landmark_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    plt.figure()
    show_landmarks(io.imread(os.path.join(folder,_path,"train",img_name)),landmark_id)

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
"""
