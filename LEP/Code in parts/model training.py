import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Set dataset path and load data
dataset_path = os.path.join(os.getcwd(), 'dataset')
loan_train = pd.read_csv(os.path.join(dataset_path, 'loan-train.csv'))
loan_test = pd.read_csv(os.path.join(dataset_path, 'loan-test.csv'))

# Data cleaning - fill missing values
loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode()[0])
loan_test['Credit_History'] = loan_test['Credit_History'].fillna(loan_test['Credit_History'].mode()[0])
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].mean())
loan_test['LoanAmount'] = loan_test['LoanAmount'].fillna(loan_test['LoanAmount'].mean())
loan_train['Gender'] = loan_train['Gender'].fillna(loan_train['Gender'].mode()[0])
loan_test['Gender'] = loan_test['Gender'].fillna(loan_test['Gender'].mode()[0])
loan_train['Dependents'] = loan_train['Dependents'].fillna(loan_train['Dependents'].mode()[0])
loan_test['Dependents'] = loan_test['Dependents'].fillna(loan_test['Dependents'].mode()[0])
loan_train['Married'] = loan_train['Married'].fillna(loan_train['Married'].mode()[0])
loan_test['Married'] = loan_test['Married'].fillna(loan_test['Married'].mode()[0])

# Encoding categorical variables
loan_train = loan_train.replace({"Loan_Status": {"Y": 1, "N": 0}, "Gender": {"Male": 1, "Female": 0},
                                 "Married": {"Yes": 1, "No": 0}, "Self_Employed": {"Yes": 1, "No": 0}})
loan_test = loan_test.replace({"Gender": {"Male": 1, "Female": 0}, "Married": {"Yes": 1, "No": 0},
                               "Self_Employed": {"Yes": 1, "No": 0}})

from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area', 'Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    loan_train[col] = le.fit_transform(loan_train[col])
    loan_test[col] = le.transform(loan_test[col])

# Feature selection and splitting data
train_features = ['Credit_History', 'Education', 'Gender']
x_train = loan_train[train_features].values
y_train = loan_train['Loan_Status'].values
x_test = loan_test[train_features].values

# Logistic Regression model training
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

# Display model parameters and accuracy
print('Coefficient of model:', logistic_model.coef_)
print('Intercept of model:', logistic_model.intercept_)

score = logistic_model.score(x_train, y_train)
print('Accuracy score overall:', score)
print('Accuracy score percent:', round(score * 100, 2))

import pickle

# Predict and map back to original labels for submission
predicted_test = logistic_model.predict(x_test)
loan_test['Loan_Status'] = predicted_test
loan_test['Loan_Status'] = loan_test['Loan_Status'].map({1: 'Y', 0: 'N'})

# Save the trained model in the same folder as the dataset
model_file_path = os.path.join(dataset_path, 'logistic_model.pkl')
pickle.dump(logistic_model, open(model_file_path, 'wb'))

# Save predictions to CSV in the same folder as the dataset
submission_file_path = os.path.join(dataset_path, 'loan_predictions.csv')
submission = loan_test[['Loan_ID', 'Loan_Status']]
submission.to_csv(submission_file_path, index=False)

print("Model and predictions saved successfully.")
