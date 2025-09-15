import numpy as np
import pandas as pd
import os

def loading_date(path="data/EMGDataset.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(base_dir)               
    file_path = os.path.join(project_root, path)  

    data = pd.read_csv(file_path, header=None)

    x = data.iloc[:, :2].values   
    y = data.iloc[:, 2].values.astype(int)    

    return x,y
