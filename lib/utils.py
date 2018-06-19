import pandas as pd
import os
import shutil
import pdb
def make_data_folder(mode):
    csv_file = "sample/"+mode+"_clean.csv"
    df=pd.read_csv(csv_file)
    print(df.head())
    directory= "sample/ImageData/"
    pdb.set_trace()
    for index,rows in df.iterrows():
        img_folder =directory+str(rows['landmark_id'])
        if not os.path.exists(img_folder):
            print("Creating Directory {}".format(img_folder))
            os.makedirs(img_folder)
        else:
            print("Folder {} exists!".format(img_folder))
        img = "sample/"+mode+"/"+str(rows["id"])+".jpg"
        print("Copying image {}".format(img))
        shutil.copy2(img,img_folder)



if __name__=="__main__":
    make_data_folder("train")
