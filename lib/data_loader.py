import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform
from torchvision import transforms, utils

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
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image,landmark_id = sample['image'], sample['landmark_id']

        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            #to ensure aspect ratio
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_h = int(new_h) , int(new_w)
        img = transform.resize(image,(new_h,new_w))
        return ({"image":img, "landmark_id":landmark_id})

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args(tuple or int): desired size, int if a square crop is desired
    """
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size

    def __call__(self,sample):
        image,landmark_id = sample['image'], sample['landmark_id']

        h,w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top+new_h,
                left:left+new_w]

        return {'image':image, 'landmark_id':landmark_id}

class ToTensor(object):
    """
    Convert ndarrays to sample Tensors
    """

    def __call__(self, sample):
        image, landmark_id = sample['image'], sample['landmark_id']
        #numpy image: H x W x C
        #torch format: C x H x W
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image),
                'landmark_id':torch.from_numpy(np.array(landmark_id))}


if __name__ == "__main__":
    landmarks_dataset = LandmarkDataset(csv_file='data/train.csv', root_dir='data/train',
           transform=transforms.Compose([
               Rescale((123,245)),
               RandomCrop(128),
               ToTensor()
               ]))
    #landmarks_dataset = LandmarkDataset(csv_file='data/train.csv', root_dir='data/train')
    """
    scale = Rescale(64)
    crop = RandomCrop((28,25))
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    fig = plt.figure()
    sample = landmarks_dataset[3]
    for i, trnsfm in enumerate([scale, crop, composed]):
        transformed_sample = trnsfm(sample)

        ax = plt.subplot(1,3,i+1)
        plt.tight_layout()
        ax.set_title(type(trnsfm).__name__)
        show_landmarks(**transformed_sample)
    plt.show()
    """
    for i in range(len(landmarks_dataset)):
        sample = landmarks_dataset[i]
        print(i, sample['image'].shape,sample['landmark_id'])

        if i==3:
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

    scale = Rescale(64)
    crop = RandomCrop((28,25))
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    fig = plt.figure()
    sample = landmarks_dataset[3]
    for i, trnsfm in enumerate([scale, crop, composed]):
        transformed_sample = trnsfm(sample)

        ax = plt.subplot(1,3,i+1)
        plt.tight_layout()
        ax.set_title(type(trnsfm).__name__)
        show_landmarks(**transformed_sample)
    plt.show()
"""
