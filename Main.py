#%% 
#import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib as plt
file = pd.read_csv('Mall_Customers.csv')
#%% 

file.plot(kind = 'density')

file.columns
#%% 
file.columns = ['customer_id', 'gender', 'age', 'annual_income', 'spending_score']
file.describe()

#%% predict if it is a male or and female 

