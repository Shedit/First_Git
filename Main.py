#%% 
#import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib as plt
file = pd.read_csv('Mall_Customers.csv')
#%% 

file.plot(kind = 'density')
#%% 

