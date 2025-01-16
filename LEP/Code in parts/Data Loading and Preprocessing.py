import os
import pandas as pd

# Suppress future downcasting warning
pd.set_option('future.no_silent_downcasting', True)

dataset_path = os.path.join(os.getcwd(), 'dataset')

# Load datasets
loan_train = pd.read_csv(os.path.join(dataset_path, 'loan-train.csv'))
loan_test = pd.read_csv(os.path.join(dataset_path, 'loan-test.csv'))

# Fill missing values
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

# Encode categorical values
loan_train = loan_train.replace({"Loan_Status": {"Y": 1, "N": 0}, "Gender": {"Male": 1, "Female": 0},
                                 "Married": {"Yes": 1, "No": 0}, "Self_Employed": {"Yes": 1, "No": 0}})
loan_test = loan_test.replace({"Gender": {"Male": 1, "Female": 0}, "Married": {"Yes": 1, "No": 0},
                               "Self_Employed": {"Yes": 1, "No": 0}})

# Label encoding for certain features
from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area', 'Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    loan_train[col] = le.fit_transform(loan_train[col])
    loan_test[col] = le.transform(loan_test[col])

print("Data loading and preprocessing completed.")
