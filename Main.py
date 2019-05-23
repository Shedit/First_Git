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
file.groupby('gender').plot(kind = 'density')
file.groupby('gender').describe().loc[:, ['annual_income']]
#%% predict if it is a male or and female 
from sklearn import preprocessing 

le = preprocessing.LabelEncoder()

file['gender'] = le.fit_transform(file['gender'])

## data and target 
cols = [c for c in file.columns if c not in ['customer_id', 'gender']]
data = file[cols]
target = file['gender']

# splits 
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
#%% Model: Gaussian Naive-bayes 

# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

#%% 
#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)
#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

#%% 
# 
# 
    
pd.plotting.scatter_matrix(file)
#%% 
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))

#%% 

# well, probably not unexpeted due to the sall set and very unspread data. More data is needed to gain some accurate predictions on whether or not it is a female or male  that is spending money. 




