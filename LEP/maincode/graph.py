import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

# Label encoding for certain columns
feature_col = ['Property_Area', 'Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    loan_train[col] = le.fit_transform(loan_train[col])
    loan_test[col] = le.transform(loan_test[col])

# Visualization: Plotting the data
loan_train.plot(figsize=(18, 8))
plt.show()

# Scatter plot: Relation Between Applicant Income vs Loan Amount
plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicant Income vs Loan Amount")
plt.scatter(loan_train['ApplicantIncome'], loan_train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.grid(True)
plt.show()

# Line plot: Loan Application Amount vs Loan Status
plt.figure(figsize=(12, 6))
plt.plot(loan_train['Loan_Status'], loan_train['LoanAmount'], marker='o', linestyle='-', color='b')
plt.title("Loan Application Amount vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.grid(True)
plt.show()

# Count plot for Loan Status
plt.figure(figsize=(12, 6))
sns.countplot(x='Loan_Status', data=loan_train)
plt.title("Loan Status Distribution")
plt.show()

# Distribution of Applicant Income
plt.figure(figsize=(12, 6))
loan_train['ApplicantIncome'].hist(bins=20)
plt.title("Distribution of Applicant Income")
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.show()

# Distribution of Loan Amount
plt.figure(figsize=(12, 6))
loan_train['LoanAmount'].hist(bins=20, color='orange')
plt.title("Distribution of Loan Amount")
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
loan_train_numeric = loan_train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(loan_train_numeric.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
plt.title("Correlation Heatmap")
plt.show()
