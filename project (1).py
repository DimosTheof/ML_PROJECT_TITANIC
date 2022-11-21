#!/usr/bin/env python
# coding: utf-8

# Machine Learning Project-D. Theofilopoulos

# In[1]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn # scikit-learn
import seaborn as sns
import random
from xgboost import XGBClassifier
from pandas import read_csv   # pandas manage "dataframes"
from pandas.plotting import scatter_matrix
from matplotlib import pyplot   # I had already imported matplotlib
from matplotlib.pyplot import scatter   # or simply use pyplot.scatter or matplotlib.pyplot.scatter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler


# In[2]:


url = "titanic_train.csv"
dataset = read_csv(url, header=0)


# Visualisation

# In[4]:


df= dataset[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].copy()
df['Sex'].replace(to_replace = ['male', 'female'], value = [0,1], inplace= True)
df['Embarked'].replace(to_replace = ['C', 'Q',"S"], value = [0,1,2], inplace= True)


# In[5]:


sns.histplot(data=df[df['Survived']==1], x="Age", color="skyblue", label="Survived", kde=True)
sns.histplot(data=df[df['Survived']==0], x="Age", color="red", label="Passed Away", kde=True)
pyplot.xlabel('Age')
#pyplot.savefig('Age_histogram.png', dpi=300)
pyplot.show()
sns.histplot(data=df[df['Survived']==1], x="Sex", color="skyblue", label="Survived", kde=True)
sns.histplot(data=df[df['Survived']==0], x="Sex", color="red", label="Passed Away", kde=True)
pyplot.xlabel('Sex')
#pyplot.savefig('Sex_histogram.png', dpi=300)
pyplot.show()
pyplot.savefig('Sex_histogram.png', dpi=300)
sns.histplot(data=df[df['Survived']==1], x="Embarked", color="skyblue", label="Survived", kde=True)
sns.histplot(data=df[df['Survived']==0], x="Embarked", color="red", label="Passed Away", kde=True)
pyplot.xlabel('Embarked')
#pyplot.savefig('Embarked.png', dpi=300)
pyplot.show()
sns.histplot(data=df[df['Survived']==1], x="SibSp", color="skyblue", label="Survived", kde=True)
sns.histplot(data=df[df['Survived']==0], x="SibSp", color="red", label="Passed Away", kde=True)
pyplot.xlabel('SibSp')
#pyplot.savefig('SibSp.png', dpi=300)
pyplot.show()
sns.histplot(data=df[df['Survived']==1], x="Pclass", color="skyblue", label="Survived", kde=True)
sns.histplot(data=df[df['Survived']==0], x="Pclass", color="red", label="Passed Away", kde=True)
pyplot.xlabel('Pclass')
#pyplot.savefig('Pclass.png', dpi=300)
pyplot.show()


# In[5]:


sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survived",height=4).map(pyplot.scatter,"Age","Sex").add_legend()
pyplot.show()


# In[6]:


sns.set_style("whitegrid");
sns.pairplot(df,hue="Survived",height=3); # height is the height in inches of each facet
pyplot.savefig('pairs.png', dpi=300) 
pyplot.show()


# In[7]:


sns.barplot(x='Pclass', y='Survived', data=df)
pyplot.savefig('Pclass_bar.png', dpi=300)


# In[4]:


sns.barplot(x='Sex', y='Survived', data=df)
pyplot.savefig('Sex_bar.png', dpi=300)


# In[9]:


sns.barplot(x='SibSp', y='Survived', data=df)#we see that when people are two the chances are higher while when persons increase chances drop


# In[6]:


df.isnull().values.any()


# In[7]:


df.isna().any()
df = df.values  


# HOLDOUT METHOD

# In[4]:


def callClassifier():
    
    X = df[:,1:]   # features(np.ndarray)
    Y = df[:,0]     # targets (classes, labels)  
    Y = Y.astype('int')
    test=0.15
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=test, random_state=1)


    df.shape

    myModel = XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
    standardizeData=True

    # fit scaler on training data
    norm = MinMaxScaler().fit(X_train)

    # transform training data
    X_train = norm.transform(X_train)

    # transform testing dataabs
    X_validation = norm.transform(X_validation)

    myModel.verbose = 0  # 1

    # Train
    myModel.fit(X_train, Y_train)

    # Calculate predictions
    Y_predicted = myModel.predict(X_validation)

    #Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(Y_predicted, Y_validation)

    #Printing the accuracy
    print("Accuracy of Classifier with threshold set to 0.5: ", cm.trace()/cm.sum())

    # ROC curves for each binary classification
    Y_probab = myModel.predict_proba(X_validation) 
    fpr2, tpr2, threshold = roc_curve(Y_validation, Y_probab[:,1])

    # Calculate the area under the roc curve
    auc = roc_auc_score(Y_validation, Y_probab[:,1])
    # Plot the ROC curve; note the varying color, the possibility to see everything on the same figure or not, the use of pause to avoid blocking, the axis limits
    # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
    # https://www.kite.com/python/answers/how-to-generate-a-random-color-for-a-matplotlib-plot-in-python#use-numpy-random-rand
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    #pyplot.figure() # start a new fig https://stackoverflow.com/questions/6916978/how-do-i-tell-matplotlib-to-create-a-second-new-plot-then-later-plot-on-the-o
    pyplot.plot(fpr2, tpr2, color = color)   # "red"
    pyplot.xlim([-0.01,1.01])
    pyplot.ylim([-0.01,1.01])
    pyplot.xlabel("FPR")
    pyplot.ylabel("TPR")
    #pyplot.draw()
    #pyplot.savefig('ROC_AUC_holdout_15per.png', dpi=300)
    pyplot.pause(0.001)
    return auc


# In[6]:


AUCs = []
for j in range(5): # 0 to n-1  (calculates the AUC n times)
     AUCs.append(callClassifier())
     print("Mean AUC is ",  numpy.mean(AUCs))
     print("AUC std is ",  numpy.std(AUCs))


# K-Fold validation

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_predict


# In[ ]:


def callClassifier_kfold():
        
    X = df[:,1:]   # features(np.ndarray)
    Y = df[:,0]     # targets (classes, labels)  
    Y = Y.astype('int')
    myModel = XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    Y_probab = cross_val_predict(myModel, X, Y, cv=kfold, method='predict_proba')
    fpr2, tpr2, threshold = roc_curve(Y, Y_probab[:,1]) # Y are the original classes

    # Calculate the area under the roc curve
    auc = roc_auc_score(Y, Y_probab[:,1])
    print("ROC-AUC of XBGClassifier: ", auc)

    # Plot the ROC curve
    pyplot.plot(fpr2, tpr2, color = "red")
    pyplot.xlim([-0.01,1.01])
    pyplot.ylim([-0.01,1.01])
    pyplot.xlabel("FPR")
    pyplot.ylabel("TPR")
    #pyplot.savefig('ROC_AUC_10fold.png', dpi=300)
    #pyplot.show()
    return auc


# In[ ]:


AUCs_kfold = []
for j in range(5): # 0 to n-1  (calculates the AUC n times)
     AUCs_kfold.append(callClassifier_kfold())
     print("Mean AUC is ",  numpy.mean(AUCs))
     print("AUC std is ",  numpy.std(AUCs))


# MLP classifier

# In[15]:


df2= dataset[["Survived","Pclass","Sex","SibSp","Parch","Fare"]].copy()
df2['Sex'].replace(to_replace = ['male', 'female'], value = [0,1], inplace= True)


# In[16]:


df2
df2.isna().any()
df2 = df2.values 


# In[17]:


X2 = df2[:,1:]   # features(np.ndarray)
Y2 = df2[:,0]     # targets (classes, labels)  
Y2 = Y2.astype('int')
test=0.15
X2_train, X2_validation, Y2_train, Y2_validation = train_test_split(X2, Y2, test_size=test, random_state=1) 
    # fit scaler on training data
norm = MinMaxScaler().fit(X2_train)

# transform training data
X2_train = norm.transform(X2_train)

# transform testing dataabs
X2_validation = norm.transform(X2_validation)


# In[18]:


def MLPClassifier():
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
    #myMLP = MLPClassifier(hidden_layer_sizes=(), max_iter=5000, activation = 'logistic', solver='sgd')
    #myMLP = MLPClassifier(hidden_layer_sizes=(), max_iter=5000, activation = 'relu', solver='adam')
    myMLP = MLPClassifier(hidden_layer_sizes=(3), max_iter=300, activation = 'relu', solver='adam')
    #myMLP = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300, activation = 'relu', solver='adam', random_state=1)
    #myMLP = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, solver='sgd', tol=1e-4,  
     #                      random_state=1,learning_rate_init=1, learning_rate='adaptive') # alpha=1e-4,
    myMLP = MLPClassifier(hidden_layer_sizes=(30,20,10,10), max_iter=1000, activation = 'relu', solver='adam')
    myMLP.verbose = 1

    # Train
    myMLP.fit(X2_train, Y2_train)

    # attributes of the trained classifier
    print('current loss computed with the loss function: ', myMLP.loss_)
    print('coefs: ', myMLP.coefs_)
    print('intercepts: ', myMLP.intercepts_)
    print('number of iterations of the solver: ', myMLP.n_iter_)
    print('num of layers: ', myMLP.n_layers_)
    print('Num of o/p: ', myMLP.n_outputs_)

    # Calculate predictions (classes, after thresholding!!)
    Y2_predicted = myMLP.predict(X2_validation)   # Y_predicted.size is 20

    #Comparing the predictions (with threshold=0.5) against the actual observations in Y_validation
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y2_predicted, Y2_validation)

    #Printing the accuracy
    print("Accuracy of MLPClassifier with threshold set to 0.5: ", cm.trace()/cm.sum())

    # ROC curves for each binary classification
    Y2_probab = myMLP.predict_proba(X2_validation)  # this is a (30,2) ndarray
    fpr2, tpr2, threshold = roc_curve(Y2_validation, Y2_probab[:,1])

    # Calculate the area under the roc curve
    print("ROC-AUC of MLPClassifier: ",roc_auc_score(Y2_validation, Y2_probab[:,1]))

    # Finally plot the ROC curve
    pyplot.plot(fpr2, tpr2, color = "red")
    pyplot.xlim([-0.01,1.01])
    pyplot.ylim([-0.01,1.01])
    pyplot.xlabel("FPR")
    pyplot.ylabel("TPR")
    pyplot.savefig('ROC_MLP.png', dpi=300)
    pyplot.show()


# In[20]:


AUCs_MLP = []
for j in range(5): # 0 to n-1  (calculates the AUC n times)
     AUCs_MLP.append(MLPClassifier())
     print("Mean AUC is ",  numpy.mean(AUCs_MLP))
     print("AUC std is ",  numpy.std(AUCs_MLP))


# In[ ]:




