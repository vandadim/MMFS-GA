
"""
Created on Thu Aug  4 10:28:30 2022

@author: vandadim

Script for evaluating a classification model using ROC plot, AUC, balanced accuracy, recall, specificity, and sensitivity.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from numpy import genfromtxt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def roc_plot(y_test,y_prob):
    """
    Plot ROC curve for binary classification.

    Args:
        y_test (array-like): True labels of the test set.
        y_prob (array-like): Predicted probabilities for the positive class.

    Returns:
        None
    """
    r_prob = [0 for _ in range(len(y_test))]
    # r_auc = roc_auc_score(y_test,r_prob)
    r_auc_fpr,r_auc_tpr, _ = roc_curve(y_test,r_prob)
    
    r_score = roc_auc_score(y_test,y_prob)
    
    r_fpr,r_tpr, _ = roc_curve(y_test,y_prob)
    plt.figure()    
    plt.plot(r_auc_fpr,r_auc_tpr, linestyle='--', label='Diagonal')

    plt.plot(r_fpr,r_tpr, linestyle='--', label='MMFS-GA, (AUC = %0.3f)' % r_score)
    

def calcAUC(Y_tst, Y_pred):
    """
    Calculate the Area Under the Curve (AUC) for each class in a multiclass classification problem.

    Args:
        Y_tst (array-like): True labels of the test set.
        Y_pred (array-like): Predicted probabilities for each class.

    Returns:
        numpy.ndarray: A 1D array containing the AUC value for each class.
    """
    n_classes = Y_pred.shape[1]
   
    # Compute ROC curve and ROC area for each class    
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros((n_classes,1))
    for i in np.arange(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_tst[:,i], Y_pred[:,i]/n_classes)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
       
    return roc_auc.flatten()
   
def multAUC(Y_tst_bin, Y_pred):
    """
    Calculate the weighted average AUC for a multiclass classification problem.

    Args:
        Y_tst_bin (array-like): True labels of the test set in binary matrix form.
        Y_pred (array-like): Predicted probabilities for each class.

    Returns:
        float: Weighted average AUC.
    """
    p_class = np.sum(Y_tst_bin,axis=0)/np.sum(Y_tst_bin)    
    return np.sum(calcAUC(Y_tst_bin,Y_pred)*p_class)  

def evaluate_function(x_train,y_train,x_test,y_test,task_type):
    """
    Evaluate the performance of a classification model.

    Args:
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        x_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        task_type (str): Type of classification task. Either 'multiclass' or 'binary'.

    Returns:
        None
    """
    if task_type == 'multiclass':
        
        
        cl_rf = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000)
        cl_rf.fit(x_train, y_train)
        y_pred=cl_rf.predict(x_test)
        y_prob=cl_rf.predict_proba(x_test)
        
        BalAcc=balanced_accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        TP=[]
        FP=[]
        TN=[]
        FN=[]
        for i in range (3):
            TP.append(cm[i][i])
            if i==0:
                FP.append((cm[1][i])+(cm[2][i]))
                TN.append((cm[1][1])+(cm[2][1])+(cm[1][2])+(cm[2][2]))
                FN.append((cm[0][1])+(cm[0][2]))
            elif i==1:
                FP.append((cm[0][i])+(cm[2][i]))
                TN.append((cm[0][0])+(cm[0][2])+(cm[2][0])+(cm[2][2]))
                FN.append((cm[1][0])+(cm[1][2]))
            elif i==2:
                FP.append((cm[0][i])+(cm[1][i]))
                TN.append((cm[0][0])+(cm[0][1])+(cm[1][0])+(cm[1][1]))
                FN.append((cm[2][0])+(cm[2][1]))
                
        Recall_1   = TP[0]/(TP[0]+ FN[0])
        Recall_2  = TP[1]/(TP[1]+ FN[1])
        Recall_3   = TP[2]/(TP[2]+ FN[2])
        # 
        y_label=np.array(pd.get_dummies(y_test))
        y_label=np.int8((y_label*2)-1)
        AUC = multAUC(y_label, y_prob)
        print("&",format(BalAcc,'.3f'),"&",format(Recall_1,'.3f'),"&",format(Recall_2,'.3f'),"&",format(Recall_3,'.3f'),"&",format(AUC,'.3f'),"\\\\")
        
        
        
    elif task_type == 'binary':
        cl_rf = LinearDiscriminantAnalysis()
        cl_rf.fit(x_train, y_train)
        y_pred=cl_rf.predict(x_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        BalAcc=balanced_accuracy_score(y_test, y_pred)
        
        # Calculate specificity and sensitivity
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        # Get predicted probabilities for test data
        y_prob=cl_rf.predict_proba(New_X_ts)[:, 1]
        
        # Calculate AUC
        AUC = roc_auc_score(y_test, y_prob)
        
        print("&",format(BalAcc,'.3f'),"&",format(sensitivity,'.3f'),"&",format(specificity,'.3f'),"&",format(AUC,'.3f'),"\\\\")

    
        
        
        
 
# input selected features
SelectedFeatures=genfromtxt('SelectedFeatures.csv', delimiter=',')   
All_Train,All_Test=[],[]
modality_number=2
for data in range (modality_number):
    Path_Train   ='D:/GeneticAlgorithm/NewExperiment/latest_Data/Data_'+str(data+1)+'_N.csv'
    Path_Test    ='D:/GeneticAlgorithm/NewExperiment/latest_Data/Data_'+str(data+1)+'_TS_N.csv'
    Data_Tr      = genfromtxt(Path_Train,delimiter=',',skip_header=1)
    Data_Ts      = genfromtxt(Path_Test,delimiter=',',skip_header=1)
    Label_Tr     = Data_Tr[:,-1]
    Label_Ts     = Data_Ts[:,-1]
    Data_X_Tr    = np.delete(Data_Tr, -1, axis=1)
    Data_X_Ts    = np.delete(Data_Ts, -1, axis=1)
    All_Train.append(Data_X_Tr)
    All_Test.append(Data_X_Ts)
X_train = np.concatenate(All_Train,axis=1)
X_test = np.concatenate(All_Test,axis=1)
con_ind = [i for (i, b) in enumerate(SelectedFeatures) if b == 1] # Find index of selected features
New_X_tr = X_train[:,con_ind] # Slected features in from all data
New_X_ts = X_test[:,con_ind]
classification_task='binary' # or multiclass


