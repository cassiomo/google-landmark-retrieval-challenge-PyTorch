import os, csv
"""
f = open("/home/pramod/work/google-landmark-retrieval-challenge-PyTorch/data/train_final_1.csv","wb")
w = csv.writer(f)
for path, dirs, files in os.walk("/home/pramod/work/google-landmark-retrieval-challenge-PyTorch/data/train/"):
        for filenames in files:
            #filenames = str(filenames)
            #name = filenames.split(".")
            w.writerow(filenames)
"""
f=open("/home/pramod/work/google-landmark-retrieval-challenge-PyTorch/sample/train_final.csv",'r+')
w=csv.writer(f)
for path, dirs, files in os.walk("/home/pramod/work/google-landmark-retrieval-challenge-PyTorch/sample/train/"):
    for filename in files:
        w.writerow([filename])
