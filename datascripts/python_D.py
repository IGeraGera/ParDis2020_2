#!/bin/python3
# Format dataset to a format that C can parse easily
# CSV with . decimal sep and , as , separator
import pandas as pd
import os
import math
import numpy as np


dataset_name='Container_Crane_Controller_Data_Set.csv'

# Get home path
HOME=os.environ['HOME']
# Join paths to final
DATA_PATH=os.path.join(HOME,'Downloads',dataset_name)
data =  pd.read_csv(DATA_PATH,sep=';',decimal=',')

dataRows=data.shape[0]
d=data.shape[1]
n = math.floor(dataRows*0.75)
m=dataRows-n

X=data[0:n].values
Y=data[n:].values

D=np.sqrt(np.multiply(X,X) @ np.ones([d,1]) @ np.ones([m,1]).T - 2*X@Y.T + np.ones([n,1])@np.ones([d,1]).T@(np.multiply(Y,Y).T))
print(D)