import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    train_size = 0.9


    if os.path.isfile("validation_split.npy"):
        raise BaseException("File exists.")

    df = pd.read_csv("training_set.txt", sep=" ", header=None, names=["source", "target", "label"])
    indexes = list(df.index)
    train, _ = train_test_split(indexes, train_size=train_size)

    validation_split = df.index.isin(train) 
    with open(f'validation_split.npy', 'wb') as f:
        np.save(f, validation_split)
    
