import pandas as pd

def make_data_folder():
    df=pd.read_csv("sample/sample_train.csv")
    folder_name = []
    for index,rows in df.iterrows():
        folder_name.append(rows['landmarks_id'])
     #   print(class_)
    return folder_name

if __name__=="__main__":
    x=make_data_folder()
    print(x)
