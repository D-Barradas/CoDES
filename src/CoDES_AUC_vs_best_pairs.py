#%%
import os
import pandas as pd
import pickle
from sklearn.metrics import  auc, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

PDB_BM5 = [
'1EXB','1JTD','1M27','1RKE','2A1A','2GAF','2GTP','2VXT','2W9E',
'2X9A','2YVJ','3A4S','3AAA','BAAD','3AAD','3BIW','3BX7',
'3DAW','3EO1','3EOA','3F1P','3FN1','3G6D','3H11',
'3H2V','3HI6','3HMX','3K75','3L5W','3L89','3LVK','3MXW',
'BP57','CP57','3P57','3PC8','3R9A','3RVW','3S9D','3SZK',
'3V6Z','3VLB','4DN4','4FQI','4FZA','4G6J','4G6M','4GAM',
'4GXU','4H03','4HX3','4IZ7','4JCV','4LW4','4M76'
]

def store(b, file_name):
    pickle.dump(b, open(file_name, "wb"))

def load(file_name):
    b = {}
    try:
        b = pickle.load(open(file_name, "rb"))
        print("Loading Successful")
        return b
    except (OSError, IOError) as e:
        print("Loading Failed. Initializing to empty")
        b = {}
        return b

def load_data_sets():
    all_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_all_data_ccharppi_4_march_2020_complete.csv",dtype={'class_q': 'object'})
    all_balanced_data.set_index('Conf',inplace=True)
    all_balanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()
    print (all_balanced_data.shape)
    all_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete.csv",dtype={'class_q': 'object'})
    all_unbalanced_data.set_index('Conf',inplace=True)
    all_unbalanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()


#     Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set_feb_12_2021.csv")
    Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set.csv")
#     Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set_march_22_2021.csv")

    print (Scorers_balanced_data.shape)


    Scorers_balanced_data.set_index('Conf',inplace=True)
    Scorers_balanced_data.dropna(inplace=True)

#     Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set_feb_12_2021.csv")
    Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set.csv")
#     Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set_march_22_2021.csv")


    Scorers_unbalanced_data.set_index('Conf',inplace=True)
    Scorers_unbalanced_data.dropna(inplace=True)

    X_train = all_balanced_data[~all_balanced_data["idx"].isin(PDB_BM5) ]
    y_train = all_balanced_data[~all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')

            ## data set for less than 5
    X_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]
    y_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')
    #         print (X_test.size,y_test.size)
            ## data set for less than 5
    X_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]
    y_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')

    X_test = Scorers_balanced_data
    y_test = Scorers_balanced_data["binary_label"].astype('bool')

    X_test_u = Scorers_unbalanced_data
    y_test_u = Scorers_unbalanced_data["binary_label"].astype('bool')

    X_test_u.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                'binary_label':'label_binary'
                            },inplace=True)
    X_test.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                   'binary_label':'label_binary'
                          },inplace=True)

#     for x in X_val_u.columns:
#         if x not in X_test.columns:
#             print (x)
    return X_train, y_train , X_val, y_val, X_test, y_test ,X_val_u, y_val_u, X_test_u, y_test_u

def scaling_data(X_train,X_test,X_test_unbalanced):
    """ As the name implies this definition scale the data
    Ideally the imputs shupold be pandas dataframes

    Args:
        X_train (dataframe): Balanced training data
        X_test (dataframe): Validation/test BALANCED data to scale according to the fit of the training
        X_test_unbalanced ([type]): Validation/test UNBALANCED data to scale

    Returns:
        scaled_train,scaled_test,scaled_test_u : scaled data
    """
#     scaler = MinMaxScaler()
    features = ['idx','class_q','pdb1','chains_pdb1','pdb2','chains_pdb2',
                'label_binary','DQ_val','binary_label','identification','labels']
    scaler = StandardScaler()

    for x in features :
        if x in X_train.columns:
            X_train= X_train.drop(x,axis=1)
    for x in features :
        if x in X_test.columns:
                X_test = X_test.drop(x,axis=1)

    for x in features :
        if x in X_test_unbalanced.columns:
            X_test_unbalanced= X_test_unbalanced.drop(x,axis=1)

    scaler.fit(X_train)

    # filename = '../models/scaler_CODES_BM4_all_features.sav'
    # pickle.dump(scaler, open(filename, 'wb'))

    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)
    scaled_test_u = scaler.transform(X_test_unbalanced)
    return scaled_train,scaled_test,scaled_test_u



def calculate_roc_and_auc(y_true, x_test ,model):
    y_score = model.predict_proba(x_test)[::,1]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr,roc_auc

names = ["Nearest Neighbors",
    "Gradient boosting",
    # "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "XgBoost",
    "SVM"
         ]
pairs_scoring_funtions = {
    0:["CP_SJKG","AP_dDFIRE"],
    1:["CP_BFKV","CP_HLPL"],
    2:["PYDOCK_TOT","CP_SKOa"],
    3:["CP_SKOa","AP_dDFIRE"],
    4:["CP_SJKG","PYDOCK_TOT"],
    5:["PYDOCK_TOT","CP_BFKV"],
    6:["PYDOCK_TOT","AP_T2"],
    7:["AP_PISA","PYDOCK_TOT"],
    8:["CP_TB","CP_BFKV"],
    9:["CP_BFKV","AP_T2"],
}

### load the data ###
X_train, y_train , X_val, y_val, X_test, y_test ,X_val_u, y_val_u, X_test_u, y_test_u  = load_data_sets()

### scale data ###
# X_train_w, X_test_w, X_test_u_w = scaling_data(X_train, X_test, X_test_u)
X_train_w, X_val_w, X_val_u_w = scaling_data(X_train, X_val, X_val_u)

#%%
def calculate_zscores(df):
    names_list , list_of_zscore = [],[]
    for i in range(10):
        name =f"{pairs_scoring_funtions[i][0]}_{pairs_scoring_funtions[i][1]}"
        try :
            list_of_zscore.append( df[pairs_scoring_funtions[i]].apply(zscore).sum(axis=1))
            names_list.append( name )
        except ValueError:
            print(f"this pair is not on df.columns : {name}")

    return list_of_zscore, names_list
#%%
def get_predicted_proba(x_val,y_true):
    my_auc = []
    my_cls_rates = {}
    for name in names:
            # print (name)
            ### is the classifer present?
            if os.path.isfile(f"../models/{name}_sklearnex.sav"):
                my_cls = load(f"../models/{name}_sklearnex.sav")
                fpr, tpr,roc_auc = calculate_roc_and_auc(y_true, x_val ,my_cls)
                my_auc.append( roc_auc )
                my_cls_rates[name]= {"fpr":fpr, "tpr":tpr}
    return my_auc, my_cls_rates

#%%
def load_CoDES(my_x,my_y):
    clf = load("../models/RFC_CODES_features_selected.sav")
    scaler = load ("../models/scaler_CODES_BM4_selected_features.sav")
    selected_feat = ['CONSRANK_val','AP_GOAP_DF','CP_TD','CP_D1','CP_HLPL','DDG_V','CP_MJ3h',
                 'PYDOCK_TOT','ELE','CP_SKOIP','SIPPER','AP_DFIRE2','AP_dDFIRE','AP_PISA','CP_RMFCA',
#                      'CP_TB','AP_DARS','CP_BT'
                ]
    my_x_val_u =  my_x[selected_feat]
    # my_x_val = scaler.transform(my_x_val)
    my_x_val_u = scaler.transform(my_x_val_u)
    fpr, tpr,roc_auc = calculate_roc_and_auc(y_true=my_y, x_test=my_x_val_u,model=clf)
    return fpr, tpr , roc_auc
#%%
def get_predicted_zscore(x_val,y_true):
    my_rates = {}
    my_auc = []
    y_score , names_list = calculate_zscores(x_val)
    df = pd.concat(y_score,axis=1)
    df.columns = names_list
    for name in names_list:
        df.sort_values(by=name,ascending=True)
        fpr, tpr, thresholds = roc_curve(y_true, df[name])
        roc_auc = auc(fpr, tpr)
        my_auc.append(roc_auc)
        my_rates[name]= {"fpr":fpr, "tpr":tpr}
    return my_auc, my_rates,names_list
# %%
my_auc, my_cls_rates = get_predicted_proba(x_val= X_val_u_w ,y_true= y_val_u )

#%%
import matplotlib.pyplot as plt
plt.figure()
ax = plt.subplot(111)
lw = 2
for n,c,a in zip ( names,my_cls_rates, my_auc) :
    print(f"{n},{a}")
    fpr , tpr =  my_cls_rates[n]["fpr"], my_cls_rates[n]["tpr"]

    ax.plot(fpr, tpr, lw=lw,label='%s (area = %0.2f)' %(n,a) )#, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title(f'{classifier} Receiver Operating Characteristic curve')
# plt.legend(loc="lower right")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("../figures/ROC_curve_Classifiers.svg",format="svg")
plt.savefig("../figures/ROC_curve_Classifiers.png",format="png")
# plt.show()
# if not os.path.exists('figures'):
#     os.makedirs('figures')
# plt.savefig(f"figures/ROC_{classifier}.svg", format='svg', dpi=1200)
plt.close()

# %%
my_auc_zscore, my_rates_zscore, names_list = get_predicted_zscore(x_val=X_val_u ,y_true= y_val_u)
# %%
plt.figure()
ax = plt.subplot(111)

lw = 2
for n,c,a in zip ( names_list,my_rates_zscore, my_auc_zscore) :
    fpr , tpr =  my_rates_zscore[n]["fpr"], my_rates_zscore[n]["tpr"]

    ax.plot(fpr, tpr, lw=lw,label='%s (area = %0.2f)' %(n,a) )#, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)

fpr,tpr,auqq = load_CoDES(my_x=X_val_u,my_y=y_val_u)
ax.plot(fpr, tpr, lw=lw,label='CoDES (area = %0.2f)' %(auqq) )#, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title(f'{classifier} Receiver Operating Characteristic curve')
# plt.legend(loc="lower right")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()
plt.tight_layout()
plt.savefig("../figures/ROC_curve_CoDES_vs_pairs.svg",format="svg")
plt.savefig("../figures/ROC_curve_CoDES_vs_pairs.png",format="png")
