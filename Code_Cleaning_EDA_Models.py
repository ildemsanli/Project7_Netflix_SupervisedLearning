#!/usr/bin/env python3

#%%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

####Data Cleaning
net=pd.read_csv('/Users/ildem/Desktop/Project_week_7/imdb_processed.csv')

net.isna().sum()

net.drop('Unnamed: 0', axis=1, inplace=True)

net.drop('composer', axis=1, inplace=True)

net.drop('writer', axis=1, inplace=True)

net.drop('cast', axis=1, inplace=True)

net[['rating', 'vote', 'runtime']]=net[['rating', 'vote', 'runtime']].apply(pd.to_numeric)

net.dtypes

net.drop(net.loc[net['kind']=='video game'].index, inplace=True)

net.drop(net.loc[net['kind']=='tv short'].index, inplace=True)


#### Grouping the kind column

net['kind']=np.where(net.kind.str.contains('movie'), 'movie', net['kind'])

net['kind']=np.where(net.kind.str.contains('series') | net.kind.str.contains('episode'), 'series', net['kind'])

net.kind.value_counts()


#### Grouping the genre column

net['genre']=net['genre'].str.replace("[", "").str.replace("]", "")

net_genre= net.genre.str.split(',').apply(pd.Series)
net['genre']=net_genre[0]

net['genre']=net['genre'].fillna('Unknown')

genre=net['genre'].value_counts()

list1=list(genre[10:].index)

for i in list1:
    net['genre']=np.where(net['genre']==i, 'Other', net['genre'])
    
net.genre.value_counts()

#### Grouping the country column

net['country']=net['country'].str.replace("[", "").str.replace("]", "")

net_country= net.country.str.split(',').apply(pd.Series)
net['country']=net_country[0]

net['country']=net['country'].fillna('Unknown')

value=net['country'].value_counts()

list2=list(value[10:].index)

for i in list2:
    net['country']=np.where(net['country']==i, 'Other', net['country'])
      
    
#### Grouping the language column

net['language']=net['language'].fillna('Unknown')

lang=net['language'].value_counts()

list3=list(lang[4:].index)

for i in list3:
    net['language']=np.where(net['language']==i, 'Other', net['language'])


net.language.value_counts()

#### Binning the rating column 

net['rating']=pd.cut(net.rating, [0, 2, 4, 6, 8, 10], labels=["very bad", "bad", "medium", "good", "very good"])

#### Scaling the vote column 

scaler = MinMaxScaler()

array_vote=np.array(net['vote']).reshape(-1, 1)

net['vote']=scaler.fit(array_vote).transform(array_vote)

#### Scaling the runtime column 

array_time=np.array(net['runtime']).reshape(-1, 1)

net['runtime']=scaler.fit(array_time).transform(array_time)


#### Dropping empty rows

net_c=net.dropna(axis=0, how='any')

net_c.shape

net_c.isna().sum()

#### Exploratory Data Analysis

# Number of movies per year 
moviesPerYear = net_c['year'].value_counts()

# Number of movies per country
moviesPerCountry = net['country'].value_counts()

# Most popular genre per year
genrePerYear = net.groupby(['genre'])['year'].value_counts().unstack().idxmax()

# Most popular genre by country
genrePerCountry = net.groupby(['genre'])['country'].value_counts().unstack().idxmax()

#### Encoding

net_c.columns

net_c.dtypes

cat=['kind', 'genre', 'rating', 'country', 'language', 'director']

net_c[cat]=net_c[cat].apply(LabelEncoder().fit_transform)

### Define target and the features

x_columns=['year', 'kind', 'genre', 'vote', 'country', 'language', 'director', 'runtime']

x=net_c[x_columns]

y=net_c['rating']


#### Splitting train and test

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)


#### RidgeClassifier

rc=RidgeClassifier().fit(x_train, y_train)
y_pred=rc.predict(x_test)
acc=rc.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

####ROC-AUC score
d = rc.decision_function(x_test)
probs = np.exp(d) / np.sum(np.exp(d))
ROC=roc_auc_score(y_test, probs, multi_class='ovr', average='macro')

print('\nRidgeClassifier\n----------\n')
print('Accuracy:', round(acc, 2))
#print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))



#%%
#### ExtraTreesClassifier

etc=ExtraTreesClassifier(n_estimators=3000, max_depth=50).fit(x_train, y_train)
y_pred=etc.predict(x_test)
acc=etc.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

####ROC-AUC score
ROC=roc_auc_score(y_test, etc.predict_proba(x_test), multi_class='ovr')

print('\nExtraTreesClassifier\n----------\n')
print('Accuracy:', round(acc, 2))
print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))


#%%
#### CategoricalNB

cnb=CategoricalNB().fit(x_train, y_train)
y_pred=cnb.predict(x_test)
acc=cnb.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####accuracy score
metrics.accuracy_score(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

####ROC-AUC score
ROC=roc_auc_score(y_test, cnb.predict_proba(x_test), multi_class='ovr')

print('\nCategoricalNB\n----------\n')
print('Accuracy:', round(acc, 2))
print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))

#%%
#### SVC

svc=SVC(probability=True).fit(x_train, y_train)
y_pred=svc.predict(x_test)
acc=svc.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####accuracy score
metrics.accuracy_score(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

####ROC-AUC score
ROC=roc_auc_score(y_test, svc.predict_proba(x_test), multi_class='ovr')

print('\nSVC metrics\n----------\n')
print('Accuracy:', round(acc, 2))
print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))

#%%
#### Feature Selection with RFE
rfe = RFE(estimator=RidgeClassifier())

m= rfe.fit(x_train, y_train)

rfe.ranking_

x.loc[:, m.support_].columns

#### Create training and test data sets with RFE results ['kind', 'vote', 'language', 'runtime']
x_train_rfe=x_train[['kind', 'vote', 'language', 'runtime']]
x_test_rfe=x_test[['kind', 'vote', 'language', 'runtime']]

####RidgeClassifier on selected features
rc=RidgeClassifier().fit(x_train_rfe, y_train)
y_pred=rc.predict(x_test_rfe)
acc=rc.score(x_test_rfe, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

#ROC=roc_auc_score(y_test, rc.decision_function(x_test_rfe), multi_class='ovr')



print('\nRidgeClassifier\n----------\n')
print('Accuracy:', round(acc, 2))
#print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))


#%%
#### Feature Selection with RFE
rfe = RFE(estimator=ExtraTreesClassifier())

m= rfe.fit(x_train, y_train)

rfe.ranking_

x.loc[:, m.support_].columns

#### Create training and test data sets with RFE results ['year', 'vote', 'director', 'runtime']
x_train_rfe=x_train[['year', 'vote', 'director', 'runtime']]
x_test_rfe=x_test[['year', 'vote', 'director', 'runtime']]

#### ExtraTreesClassifier

etc=ExtraTreesClassifier(n_estimators=3000, max_depth=50).fit(x_train_rfe, y_train)
y_pred=etc.predict(x_test_rfe)
acc=etc.score(x_test_rfe, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')

ROC=roc_auc_score(y_test, etc.predict_proba(x_test_rfe), multi_class='ovr')

print('\nExtraTreesClassifier\n----------\n')
print('Accuracy:', round(acc, 2))
print('ROC-AUC score:', round(ROC, 2))
print('Confusion matrix\n', matrix)
print('Precision score:', round(p, 2))
print('Recall score:', round(r, 2))
print('F1 score:', round(f1, 2))




