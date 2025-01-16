import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (make sure the path is correctly set up)
dataset_path = os.path.join(os.getcwd(), 'dataset')
loan_train = pd.read_csv(os.path.join(dataset_path, 'loan-train.csv'))

# Fill missing values as in preprocessing
loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode()[0])
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].mean())
loan_train['Gender'] = loan_train['Gender'].fillna(loan_train['Gender'].mode()[0])
loan_train['Dependents'] = loan_train['Dependents'].fillna(loan_train['Dependents'].mode()[0])
loan_train['Married'] = loan_train['Married'].fillna(loan_train['Married'].mode()[0])

# Data visualization

# General plot
loan_train.plot(figsize=(18, 8))
plt.show()

# Scatter plot
plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicant Income vs Loan Amount")
plt.scatter(loan_train['ApplicantIncome'], loan_train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.grid(True)
plt.show()

# Line plot for Loan Status vs Loan Amount
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

import numpy as np

# Histogram for Applicant Income
plt.figure(figsize=(12, 6))
loan_train['ApplicantIncome'].hist(bins=20)
plt.title("Distribution of Applicant Income")
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.show()

# Histogram for Loan Amount
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
