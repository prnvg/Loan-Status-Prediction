import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("train.csv")

print(df['ApplicantIncome'].hist(bins=50))
print(df.boxplot(column='ApplicantIncome'))
print(df.boxplot(column='ApplicantIncome', by = 'Education'))
print(df.boxplot(column='ApplicantIncome', by = 'Gender'))
print(df['LoanAmount'].hist(bins=50))
print(df.boxplot(column='LoanAmount'))
