#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os ,sys 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from operator import itemgetter
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict , cross_val_score ,KFold
import math
import pandas as pd 


# In[ ]:


PDB_BM5 = [
'1EXB','1JTD','1M27','1RKE','2A1A','2GAF','2GTP','2VXT','2W9E',
'2X9A','2YVJ','3A4S','3AAA','BAAD','3AAD','3BIW','3BX7',
'3DAW','3EO1','3EOA','3F1P','3FN1','3G6D','3H11',
'3H2V','3HI6','3HMX','3K75','3L5W','3L89','3LVK','3MXW',
'BP57','CP57','3P57','3PC8','3R9A','3RVW','3S9D','3SZK',
'3V6Z','3VLB','4DN4','4FQI','4FZA','4G6J','4G6M','4GAM',
'4GXU','4H03','4HX3','4IZ7','4JCV','4LW4','4M76'
]

PDB_score_set_target =[
    'Target39', 'Target41','Target50', 'Target53'
    
]


# In[ ]:


print(f"pandas version:{pd.__version__}")
print(f"numpy version:{np.__version__}")
# print(f"sklearn version:{sklearn.__version__}")

np.random.seed(101)


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


# In[ ]:


def load_data_sets():
    all_balanced_data = dd.read_csv("../data/Clean_dataframe_balanced_all_data_ccharppi_4_march_2020_complete.csv")
    all_balanced_data = all_balanced_data.drop("class_q",axis=1)
    all_balanced_data = all_balanced_data.set_index('Conf')
#     print (all_balanced_data.shape)
#     all_balanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()
    
#     all_unbalanced_data = dd.read_csv("../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete.csv",dtype={'class_q': 'object'})
#     all_unbalanced_data = all_unbalanced_data.drop("class_q",axis=1)
#     all_unbalanced_data = all_unbalanced_data.set_index('Conf')
#     all_unbalanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()


    Scorers_balanced_data = dd.read_csv("../data/Clean_dataframe_balanced_scorers_set_march_22_2021.csv")
    Scorers_balanced_data = Scorers_balanced_data.set_index('Conf')
    Scorers_balanced_data = Scorers_balanced_data.dropna()
    Scorers_balanced_data = Scorers_balanced_data[~Scorers_balanced_data["idx"].isin(PDB_score_set_target)]

    Scorers_unbalanced_data = dd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set_march_22_2021.csv")
    Scorers_unbalanced_data = Scorers_unbalanced_data.set_index('Conf')
    Scorers_unbalanced_data = Scorers_unbalanced_data.dropna()
#     Scorers_unbalanced_data = Scorers_unbalanced_data[~Scorers_unbalanced_data["idx"].isin(PDB_score_set_target) ]

    
    Scorers_unbalanced_data = Scorers_unbalanced_data.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                'binary_label':'label_binary'
                            })
    Scorers_balanced_data= Scorers_balanced_data.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                   'binary_label':'label_binary'
                          })
    all_balanced_data = dd.concat([all_balanced_data,Scorers_balanced_data[all_balanced_data.columns]])
#     all_unbalanced_data = dd.concat([all_unbalanced_data,Scorers_unbalanced_data[all_balanced_data.columns]])
#     all_unbalanced_data = all_unbalanced_data[~all_unbalanced_data["idx"].isin(PDB_score_set_target)]


    X_train = all_balanced_data
    y_train = all_balanced_data["DQ_val"]
#     X_train = all_unbalanced_data
#     y_train = all_unbalanced_data["DQ_val"]


#     y_train = all_balanced_data[~all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')

            ## data set for less than 5 
#     X_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]
#     y_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]["DQ_val"]

#     y_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')
    #         print (X_test.size,y_test.size)
            ## data set for less than 5 
#     X_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]
#     y_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]["DQ_val"]
#     y_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')

    
#     X_test = Scorers_balanced_data
#     y_test = Scorers_balanced_data["DQ_val"]
    
    X_test_u = Scorers_unbalanced_data[Scorers_unbalanced_data["idx"].isin(PDB_score_set_target)]
    y_test_u = Scorers_unbalanced_data[Scorers_unbalanced_data["idx"].isin(PDB_score_set_target) ]["DQ_val"]
#     y_test_u = Scorers_unbalanced_data["binary_label"].astype('bool')

    
#     X_test_u = X_test_u.rename(columns={'NIS Polar' :'Nis_Polar',
#                                   'Nis Apolar':'Nis_Apolar',
#                                   'BSA Apolar':'BSA_Apolar',
#                                   'BSA Polar' :'BSA_Polar',
#                                 'binary_label':'label_binary'
#                             })
#     X_test= X_test.rename(columns={'NIS Polar' :'Nis_Polar',
#                                   'Nis Apolar':'Nis_Apolar',
#                                   'BSA Apolar':'BSA_Apolar',
#                                   'BSA Polar' :'BSA_Polar',
#                                    'binary_label':'label_binary'
#                           })
    
#     for x in X_val_u.columns:
#         if x not in X_test.columns:
#             print (x)
    return X_train, y_train , X_test_u, y_test_u 


# In[ ]:


def scaling_data(X_train,X_test_unbalanced):
#     scaler = MinMaxScaler()
    features = ['idx','class_q','pdb1','chains_pdb1','pdb2','chains_pdb2',
                'label_binary','DQ_val','binary_label','identification','labels']
    scaler = StandardScaler()



    for x in features :
        if x in X_train.columns:
            X_train= X_train.drop(x,axis=1)            
#     for x in features :
#         if x in X_test.columns: 
#                 X_test = X_test.drop(x,axis=1)
            
    for x in features :
        if x in X_test_unbalanced.columns:
            X_test_unbalanced= X_test_unbalanced.drop(x,axis=1)
            
        
#     print(X_train.shape)
#     print(X_test.shape)

    scaler.fit(X_train)
    scaled_train = scaler.transform(X_train)
#     scaled_test = scaler.transform(X_test[X_train.columns])
    scaled_test_u = scaler.transform(X_test_unbalanced[X_train.columns])
    return scaled_train,scaled_test_u


# In[ ]:


def regression_report (y_true, y_pred ,name) :
    

    r=r2_score(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred)
    mse=mean_squared_error(y_true, y_pred)
    
    print ("R^2:",r)
    print ("MAE:",mae)
    print ("MSE:",mse)
    


# In[ ]:


X_train, y_train ,  X_test_u, y_test_u  = load_data_sets()


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 220, num = 22)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8 ]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


X_train_fs, X_test_u_fs = scaling_data(X_train, X_test_u)


# In[ ]:


base_model = RandomForestRegressor(n_estimators = 10, random_state = 101)
base_model.fit(X_train_fs, y_train)


# In[ ]:


cv = KFold(10, shuffle=True)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 100, cv = cv, verbose=2, random_state=101, n_jobs = -1)


# In[ ]:


rf_random.fit(X_train,y_train )

print ("Best params")
print (rf_random.best_params_)
print ("++++++++++++++")


# In[ ]:


base_accuracy = evaluate(base_model, X_test, y_test)


predictions = base_model.predict(X_test)


print ("base model")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test )

predictions = rf_random.predict(X_test)
print ("best RFR")
print ("R^2:",r2_score(y_test, predictions))
print ("MAE:",mean_absolute_error(y_test, predictions))
print ("MSE:",mean_squared_error(y_test, predictions))

# In[ ]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

