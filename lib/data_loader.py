import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg
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
