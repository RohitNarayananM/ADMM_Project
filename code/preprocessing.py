from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


pollution_df=pd.read_csv('datasets/pollution.csv')
student_df = pd.read_csv('datasets/StudentsPerformance.csv')


pollution_df=pollution_df.sample(n=1000)
pollution_df=pollution_df.drop(['date'], axis=1)

#Making the data numeric

labelencoder = LabelEncoder()
student_df['gender'] = labelencoder.fit_transform(student_df['gender'])
student_df['ethnicity'] = labelencoder.fit_transform(student_df['ethnicity'])
student_df['parental level of education'] = labelencoder.fit_transform(
    student_df['parental level of education'])
student_df['lunch'] = labelencoder.fit_transform(student_df['lunch'])
student_df['test preparation course'] = labelencoder.fit_transform(
    student_df['test preparation course'])

pollution_df['wnd_dir'] = labelencoder.fit_transform(pollution_df['wnd_dir'])

pollution_train, pollution_test = train_test_split(pollution_df, test_size=0.2, random_state=110)
student_train, student_test = train_test_split(student_df, test_size=0.2, random_state=110)

student_X_train = student_train.drop(["writing score"], axis=1)
student_Y_train = student_train["writing score"]
student_X_test = student_test.drop(["writing score"], axis=1)
student_Y_test = student_test["writing score"]

pollution_X_train = pollution_train.drop(['pollution'], axis=1)
pollution_Y_train = pollution_train['pollution']
pollution_X_test = pollution_test.drop(['pollution'], axis=1)
pollution_Y_test = pollution_test['pollution']

student_X_train = np.array(student_X_train)
student_Y_train = np.array(student_Y_train)
student_X_test = np.array(student_X_test)
student_Y_test = np.array(student_Y_test)
pollution_X_train = np.array(pollution_X_train)
pollution_Y_train = np.array(pollution_Y_train)
pollution_X_test = np.array(pollution_X_test)
pollution_Y_test = np.array(pollution_Y_test)

if __name__=="__main__":
    print("Student Data")
    print(student_X_train.shape, student_Y_train.shape,student_X_test.shape, student_Y_test.shape)
    print("Pollution Data")
    print(pollution_X_train.shape, pollution_Y_train.shape,pollution_X_test.shape, pollution_Y_test.shape)
