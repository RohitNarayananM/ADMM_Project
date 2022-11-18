from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

fraud_df=pd.read_csv('../datasets/Fraud.csv')

#Making the data numeric

labelencoder = LabelEncoder()

fraud_df['type']=labelencoder.fit_transform(fraud_df['type'])
fraud_df['nameOrig']=labelencoder.fit_transform(fraud_df['nameOrig'])
fraud_df['nameDest']=labelencoder.fit_transform(fraud_df['nameDest'])

fraud_train, fraud_test = train_test_split(fraud_df, test_size=0.2, random_state=110)

fraud_X_train = fraud_train.drop(['isFraud'], axis=1)
fraud_Y_train = fraud_train['isFraud']
fraud_X_test = fraud_test.drop(['isFraud'], axis=1)
fraud_Y_test = fraud_test['isFraud']

fraud_X_train = np.array(fraud_X_train)
fraud_Y_train = np.array(fraud_Y_train)
fraud_X_test = np.array(fraud_X_test)
fraud_Y_test = np.array(fraud_Y_test)


if __name__=="__main__":
    print("Fraud Data")
    print(fraud_X_train.shape, fraud_Y_train.shape,fraud_X_test.shape, fraud_Y_test.shape)
    
