# Format dataset to a format that C can parse easily
# CSV with . decimal sep and , as , separator
import pandas as pd
import os
import numpy


dataset_name='MiniBooNE_PID.txt'

# Get home path
HOME=os.environ['HOME']
# Join paths to final
DATA_PATH=os.path.join(HOME,'Downloads',dataset_name)
data =  pd.read_csv(DATA_PATH)
#data =  pd.read_csv(DATA_PATH,sep=' ',decimal='.')
data = data.values
#numpy.savetxt(dataset_name,data[:,:-1],delimiter=",")
#data.to_csv(dataset_name,index=False,header=False)
pd.DataFrame(data[:,:]).to_csv(dataset_name,index=False,header=False)
