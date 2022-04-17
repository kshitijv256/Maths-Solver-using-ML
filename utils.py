'''
Defined some functions for easy access in other files
Planning to add more functions when needed
written on 3/28/2022 by kshitijv256

'''

import pickle
import pandas as pd

def load(path):
    print('Loading data...')
    with open(path) as f:
        return pd.read_csv(f)

def pick(path):
    print('Loading Model...')
    with open(path,'rb') as f:
        return pickle.load(f)

def dump(model,path):
    with open(path,'wb') as f:
        pickle.dump(model,f)