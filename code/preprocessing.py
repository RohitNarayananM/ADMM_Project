from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd


heart_df = pd.read_csv('../datasets/heart.csv')
student_df = pd.read_csv('../datasets/StudentsPerformance.csv')

#Making the data numeric

labelencoder = LabelEncoder()
student_df['gender'] = labelencoder.fit_transform(student_df['gender'])
student_df['ethnicity'] = labelencoder.fit_transform(student_df['ethnicity'])
student_df['parental level of education'] = labelencoder.fit_transform(
    student_df['parental level of education'])
student_df['lunch'] = labelencoder.fit_transform(student_df['lunch'])
student_df['test preparation course'] = labelencoder.fit_transform(
    student_df['test preparation course'])

heart_train, heart_test = train_test_split(heart_df, test_size=0.2, random_state=110)
student_train, student_test = train_test_split(student_df, test_size=0.2, random_state=110)


heart_X_train = heart_train.drop(['target'], axis=1)
heart_Y_train = heart_train['target']
heart_X_test = heart_test.drop(['target'], axis=1)
heart_Y_test = heart_test['target']

student_X_train = student_train.drop(["writing score"], axis=1)
student_Y_train = student_train["writing score"]
student_X_test = student_test.drop(["writing score"], axis=1)
student_Y_test = student_test["writing score"]

if __name__=="__main__":
    print(student_X_train.shape, student_Y_train.shape,
        student_X_test.shape, student_Y_test.shape)
