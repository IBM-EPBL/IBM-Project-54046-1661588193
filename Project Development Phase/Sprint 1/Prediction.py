import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df=pd.read_csv('forestfires.csv')

df=df.drop(['X','Y','month','day'],axis=1)

def preprocessing(df,task):
    df=df.copy()
    if task=='Regression':
        Y=df['area']
    elif task=='Classification':
        Y=df['area'].apply(lambda x: 1 if x>0 else 0)

    X=df.drop('area',axis=1)

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.60,shuffle=True,random_state=0)

    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train=pd.DataFrame(scaler.transform(X_train),columns=X.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)

    return X_train,X_test,Y_train,Y_test


X_train,X_test,Y_train,Y_test=preprocessing(df,task='Classification')


nn_classifier_model=MLPClassifier(activation='relu',hidden_layer_sizes=(16,16),n_iter_no_change=100,solver='adam')
nn_classifier_model.fit(X_train,Y_train)

model=joblib.dump(nn_classifier_model,'forestfiremodel.pkl')








